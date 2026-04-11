//! WASM target commands: `rayzor run --wasm` and `rayzor build --target wasm`.
//!
//! Generates WASI core modules / Component Model output, links against the
//! pre-built runtime-wasm library, runs inside embedded wasmtime, and emits
//! an ES6 harness plus thread/worker runtime scripts for browser deploy.

use std::path::PathBuf;

use crate::compile_helpers::{compile_haxe_to_mir, compile_haxe_to_mir_with_defines};


pub fn cmd_run_wasm(
    file: Option<PathBuf>,
    _rpkg_files: Vec<PathBuf>,
    safety_warnings: bool,
) -> Result<(), String> {
    let file = file.ok_or_else(|| "file path required for --wasm".to_string())?;
    let source = std::fs::read_to_string(&file)
        .map_err(|e| format!("failed to read {}: {}", file.display(), e))?;

    eprintln!("Compiling {} [wasm]...", file.display());

    // Resolve workspace class-paths from rayzor.toml
    let extra_source_dirs: Vec<PathBuf> = {
        let file_dir = file.parent().and_then(|p| {
            let abs = if p.is_absolute() {
                p.to_path_buf()
            } else {
                std::env::current_dir().unwrap_or_default().join(p)
            };
            compiler::workspace::find_project_root(&abs)
        });
        file_dir
            .and_then(|root| compiler::workspace::load_project(&root).ok())
            .map(|p| p.resolved_class_paths())
            .unwrap_or_default()
    };

    // Compile Haxe → MIR → WASM
    let (mir_module, _diagnostics) = compile_haxe_to_mir(
        &source,
        file.to_str().unwrap_or("unknown"),
        Vec::new(),
        &extra_source_dirs,
        safety_warnings,
    )?;

    let user_wasm =
        compiler::codegen::wasm_backend::WasmBackend::compile(&[&mir_module], Some("main"))?;

    // Link with runtime
    let runtime_path = find_wasm_runtime();
    let linked_wasm = if let Some(rt_path) = &runtime_path {
        let rt_bytes =
            std::fs::read(rt_path).map_err(|e| format!("failed to read runtime: {}", e))?;
        compiler::codegen::wasm_linker::WasmLinker::link(&user_wasm, &rt_bytes)?
    } else {
        user_wasm
    };

    eprintln!("Running ({:.1} KB)...", linked_wasm.len() as f64 / 1024.0);

    // Execute via embedded wasmtime
    compiler::codegen::wasm_runner::run_wasm(&linked_wasm)
}

pub fn cmd_build_wasm(
    file: Option<PathBuf>,
    output: Option<PathBuf>,
    target: String,
    browser: bool,
) -> Result<(), String> {
    let file = file.ok_or_else(|| "file path required for WASM build".to_string())?;
    let source = std::fs::read_to_string(&file)
        .map_err(|e| format!("failed to read {}: {}", file.display(), e))?;

    // Resolve workspace project for class-paths and default output directory
    let project = {
        let file_dir = file.parent().and_then(|p| {
            let abs = if p.is_absolute() {
                p.to_path_buf()
            } else {
                std::env::current_dir().unwrap_or_default().join(p)
            };
            compiler::workspace::find_project_root(&abs)
        });
        file_dir.and_then(|root| compiler::workspace::load_project(&root).ok())
    };

    let extra_source_dirs: Vec<PathBuf> = project
        .as_ref()
        .map(|p| p.resolved_class_paths())
        .unwrap_or_default();

    // Default output: .rayzor/build/<name>.wasm (relative to project root or cwd)
    let output = output.or_else(|| {
        let stem = file.file_stem()?.to_string_lossy().to_string();
        let build_dir = project
            .as_ref()
            .map(|p| p.root.join(".rayzor/build"))
            .unwrap_or_else(|| PathBuf::from(".rayzor/build"));
        Some(build_dir.join(format!("{}.wasm", stem)))
    });

    println!("Building {} [target: {}]...", file.display(), target);

    // Use the full compile pipeline with "wasm" define for conditional compilation
    let mir_result = compile_haxe_to_mir_with_defines(
        &source,
        file.to_str().unwrap_or("unknown"),
        Vec::new(),
        &extra_source_dirs,
        false,
        &["wasm"],
    )?;

    let user_wasm = compiler::codegen::wasm_backend::WasmBackend::compile_with_method_map(
        &[&mir_result.module],
        Some("main"),
        &mir_result.qualified_method_map,
    )?;
    let _ = std::fs::write("/tmp/rayzor_prelink.wasm", &user_wasm);

    // Build host function map from rayzor.toml [wasm] hosts:
    // Scan each JS host file for `export function` names and map them to the module name.
    let host_functions: std::collections::BTreeMap<String, String> = {
        let config_hosts = project
            .as_ref()
            .map(|p| p.resolved_wasm_hosts())
            .unwrap_or_default();
        let mut map = std::collections::BTreeMap::new();
        for (module_name, js_path) in &config_hosts {
            if let Ok(js_source) = std::fs::read_to_string(js_path) {
                let exports = compiler::codegen::wasm_linker::WasmLinker::scan_js_exports(&js_source);
                println!("  host: {} ({} exports from {})", module_name, exports.len(), js_path.display());
                for name in exports {
                    map.insert(name, module_name.clone());
                }
            }
        }
        map
    };

    // Link with pre-built WASM runtime (if available)
    let runtime_wasm_path = find_wasm_runtime();
    let linked_wasm = if let Some(rt_path) = &runtime_wasm_path {
        let rt_bytes =
            std::fs::read(rt_path).map_err(|e| format!("failed to read runtime WASM: {}", e))?;
        println!(
            "  linking with {} ({:.1} KB)",
            rt_path.display(),
            rt_bytes.len() as f64 / 1024.0
        );
        compiler::codegen::wasm_linker::WasmLinker::link_with_hosts(&user_wasm, &rt_bytes, &host_functions)?
    } else {
        println!("  warning: WASM runtime not found, output needs JS harness");
        user_wasm
    };

    let out_path = output.unwrap_or_else(|| file.with_extension("wasm"));
    // Ensure output directory exists
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Write core module first (JS/browser use — WebAssembly.instantiate needs core modules)
    let core_path = out_path.with_extension("core.wasm");
    std::fs::write(&core_path, &linked_wasm)
        .map_err(|e| format!("failed to write {}: {}", core_path.display(), e))?;

    // Wrap as WASI P2 Component (non-fatal for browser builds with host imports)
    match compiler::codegen::wasm_component::wrap_as_component(
        &linked_wasm,
        compiler::codegen::wasm_component::ComponentKind::Command,
    ) {
        Ok(component_bytes) => {
            println!(
                "  component: {:.1} KB (core: {:.1} KB)",
                component_bytes.len() as f64 / 1024.0,
                linked_wasm.len() as f64 / 1024.0
            );
            std::fs::write(&out_path, &component_bytes)
                .map_err(|e| format!("failed to write {}: {}", out_path.display(), e))?;
        }
        Err(e) => {
            if browser {
                // Browser builds don't need Component Model — core module is sufficient
                println!("  note: component encoding skipped ({})", e);
                // Write core module as .wasm too for compatibility
                std::fs::write(&out_path, &linked_wasm)
                    .map_err(|err| format!("failed to write {}: {}", out_path.display(), err))?;
            } else {
                return Err(format!("Failed to encode WASM Component: {}", e));
            }
        }
    }

    // Collect exported class metadata for JS bindings
    let modules_ref = [&mir_result.module];
    let exported_classes = compiler::codegen::wasm_bindgen::collect_exported_classes(
        &modules_ref,
        &mir_result.class_alloc_sizes,
    );

    // Build js_imports from host_functions map (grouped by module name)
    let mut js_imports: std::collections::BTreeMap<String, Vec<(String, String)>> =
        std::collections::BTreeMap::new();
    for (func_name, module_name) in &host_functions {
        js_imports
            .entry(module_name.clone())
            .or_default()
            .push((func_name.clone(), func_name.clone()));
    }

    // Resolve and copy host modules to build output directory
    let build_dir = out_path.parent().unwrap_or(std::path::Path::new("."));
    let resolved_hosts = resolve_and_copy_host_modules(&js_imports, &project, build_dir);

    // Stage the rayzor thread runtime + worker script into the build dir so the
    // generated JS harness can import them. These files are shipped alongside
    // the compiler and provide the Web-Worker-backed Thread/Channel/Mutex
    // implementation used in the browser (`rayzor_threads.js` +
    // `rayzor_worker.js`).
    let thread_runtime_src: &str =
        include_str!("../compiler/data/rayzor_threads.js");
    let thread_worker_src: &str =
        include_str!("../compiler/data/rayzor_worker.js");
    let thread_runtime_path = build_dir.join("rayzor_threads.js");
    let thread_worker_path = build_dir.join("rayzor_worker.js");
    if let Err(e) = std::fs::write(&thread_runtime_path, thread_runtime_src) {
        eprintln!(
            "  warning: failed to stage rayzor_threads.js: {} ({})",
            e,
            thread_runtime_path.display()
        );
    }
    if let Err(e) = std::fs::write(&thread_worker_path, thread_worker_src) {
        eprintln!(
            "  warning: failed to stage rayzor_worker.js: {} ({})",
            e,
            thread_worker_path.display()
        );
    }

    // Generate JS bindings (ES6 module with class wrappers if @:export classes exist)
    let js_path = out_path.with_extension("js");
    // JS bindings use the core module (not the Component — browsers can't instantiate Components yet)
    let core_wasm_filename = core_path.file_name().unwrap().to_string_lossy();
    let js_content = if !exported_classes.is_empty() {
        // Generate ES6 class wrappers
        println!(
            "  exports: {}",
            exported_classes
                .iter()
                .map(|c| {
                    let methods: Vec<&str> = c
                        .instance_methods
                        .iter()
                        .map(|m| m.name.as_str())
                        .chain(c.static_methods.iter().map(|m| m.name.as_str()))
                        .collect();
                    format!("{} ({})", c.name, methods.join(", "))
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
        compiler::codegen::wasm_bindgen::generate_es6_bindings(
            &exported_classes,
            &core_wasm_filename,
        )
    } else {
        // Fall back to basic JS harness (with host module auto-wiring)
        generate_wasm_js_harness(&core_wasm_filename, &js_imports, &resolved_hosts)
    };
    std::fs::write(&js_path, &js_content)
        .map_err(|e| format!("failed to write {}: {}", js_path.display(), e))?;

    // Generate .d.ts and .wit if there are exported classes
    if !exported_classes.is_empty() {
        let dts_path = out_path.with_extension("d.ts");
        let dts = compiler::codegen::wasm_bindgen::generate_typescript_defs(&exported_classes);
        std::fs::write(&dts_path, &dts)
            .map_err(|e| format!("failed to write {}: {}", dts_path.display(), e))?;

        // Generate WIT (WebAssembly Interface Types) for Component Model interop
        let wit_path = out_path.with_extension("wit");
        let package_name = out_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "module".to_string());
        let wit = compiler::codegen::wasm_wit::generate_wit(&package_name, &exported_classes);
        std::fs::write(&wit_path, &wit)
            .map_err(|e| format!("failed to write {}: {}", wit_path.display(), e))?;
    }

    // Generate HTML if --browser
    if browser {
        let html_path = out_path.with_extension("html");
        let js_name = js_path.file_name().unwrap().to_string_lossy();
        let title = file.file_stem().unwrap().to_string_lossy();
        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
    #output {{ white-space: pre-wrap; background: #16213e; padding: 16px; border-radius: 8px; }}
  </style>
</head>
<body>
  <script type="module" src="{js_name}"></script>
</body>
</html>"#,
            title = title,
            js_name = js_name,
        );
        std::fs::write(&html_path, &html)
            .map_err(|e| format!("failed to write {}: {}", html_path.display(), e))?;
        println!(
            "  wrote {} ({:.1} KB)",
            out_path.display(),
            linked_wasm.len() as f64 / 1024.0,
        );
        println!("  wrote {}", js_path.display());
        println!("  wrote {}", html_path.display());
        println!(
            "\n  Serve: npx serve . && open http://localhost:3000/{}",
            html_path.file_name().unwrap().to_string_lossy()
        );
    } else {
        println!(
            "  wrote {} ({:.1} KB)",
            out_path.display(),
            linked_wasm.len() as f64 / 1024.0,
        );
        if runtime_wasm_path.is_some() {
            println!("\n  Run: wasmtime {}", out_path.display());
        }
        println!("  Run: node {}", js_path.display());
    }

    Ok(())
}

/// Find the pre-built WASM runtime library.
/// Searches: ./runtime-wasm/target/..., ~/.rayzor/lib/, alongside the rayzor binary.
pub fn find_wasm_runtime() -> Option<PathBuf> {
    let mut candidates = vec![
        // Development: built from source (cwd-relative)
        PathBuf::from("runtime-wasm/target/wasm32-wasip1/release/rayzor_runtime_wasm.wasm"),
    ];

    // Development: relative to the rayzor binary (works from any cwd)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            // binary is in target/release/, repo root is ../../
            let repo_root = bin_dir
                .join("../../runtime-wasm/target/wasm32-wasip1/release/rayzor_runtime_wasm.wasm");
            candidates.push(repo_root);
            // Installed: alongside rayzor binary
            candidates.push(bin_dir.join("rayzor_runtime_wasm.wasm"));
        }
    }

    // User home
    candidates.push(dirs_next().join("lib/rayzor_runtime_wasm.wasm"));

    candidates.into_iter().find(|p| p.exists())
}

pub fn dirs_next() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".rayzor"))
        .unwrap_or_else(|_| PathBuf::from(".rayzor"))
}

/// Resolve host module files for @:jsImport modules and copy them to the build directory.
/// Returns a map of module_name → relative filename in build dir (for JS import statements).
pub fn resolve_and_copy_host_modules(
    js_imports: &std::collections::BTreeMap<String, Vec<(String, String)>>,
    project: &Option<compiler::workspace::Project>,
    build_dir: &std::path::Path,
) -> std::collections::BTreeMap<String, String> {
    let mut resolved: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();

    if js_imports.is_empty() {
        return resolved;
    }

    // Collect host paths from rayzor.toml [wasm] hosts
    let config_hosts: std::collections::BTreeMap<String, PathBuf> = project
        .as_ref()
        .map(|p| p.resolved_wasm_hosts())
        .unwrap_or_default();

    for module_name in js_imports.keys() {
        // Try rayzor.toml config first
        if let Some(host_path) = config_hosts.get(module_name) {
            if host_path.exists() {
                let js_filename = host_path.file_name().unwrap().to_string_lossy().to_string();
                let dest = build_dir.join(&js_filename);
                if let Err(e) = std::fs::copy(host_path, &dest) {
                    eprintln!("  warning: failed to copy host {}: {}", host_path.display(), e);
                    continue;
                }

                // Also copy companion .wasm file if it exists (wasm-bindgen output)
                let bg_wasm = host_path.with_file_name(
                    host_path.file_stem().unwrap().to_string_lossy().to_string() + "_bg.wasm"
                );
                if bg_wasm.exists() {
                    let bg_dest = build_dir.join(bg_wasm.file_name().unwrap());
                    let _ = std::fs::copy(&bg_wasm, &bg_dest);
                }

                println!("  host: {} → {}", module_name, js_filename);
                resolved.insert(module_name.clone(), js_filename);
            }
        }
        // TODO: also check rpkg JsHost entries
    }

    resolved
}

pub fn generate_wasm_js_harness(
    wasm_filename: &str,
    _js_imports: &std::collections::BTreeMap<String, Vec<(String, String)>>,
    resolved_hosts: &std::collections::BTreeMap<String, String>,
) -> String {
    // Generate import statements for resolved host modules
    let mut host_imports = String::new();
    let mut host_init = String::new();
    let mut host_import_objects = String::new();
    for (module_name, filename) in resolved_hosts {
        let var_name = module_name.replace('-', "_");
        host_imports.push_str(&format!(
            "import __{var} from './{filename}';\nimport * as _{var} from './{filename}';\n",
            var = var_name,
            filename = filename,
        ));
        // Init wasm-bindgen module (loads _bg.wasm), then pre-call async exports
        // so they're resolved before WASM starts (WASM can't await Promises).
        host_init.push_str(&format!(
            "  try {{\n    await __{var}();\n    await _precall_async(_{var});\n    console.log('[rayzor] {module} host initialized (' + Object.keys(_{var}).filter(k => typeof _{var}[k] === 'function').length + ' functions)');\n  }} catch(e) {{ console.warn('[rayzor] {module} init failed:', e); }}\n",
            var = var_name,
            module = module_name,
        ));
        // Build import namespace for this host module.
        // The WASM module imports from e.g. "rayzor-gpu" module name.
        // String/byte parameters need adaptation: Rayzor passes i32 pointers into its
        // own linear memory, but wasm-bindgen expects JS strings/Uint8Arrays.
        // We generate wrappers that read from Rayzor's memory before calling the export.
        host_import_objects.push_str(&format!(
            "    '{module}': _make_host_adapter(_{var}),\n",
            module = module_name,
            var = var_name,
        ));
    }
    format!(
        r#"// Rayzor WASM Runtime Harness
// Generated by `rayzor build --target wasm`
// Run: node {filename}  |  Open in browser with a local server

{host_imports}
// Rayzor thread runtime (Web Worker pool + Atomics-based sync primitives).
// Only loaded in the browser — Node.js falls back to the inline Atomics stubs.
import {{ RayzorThreadRuntime }} from './rayzor_threads.js';
const isNode = typeof process !== "undefined" && process.versions?.node;

// Linear memory (shared with WASM module)
let memory;

// Simple bump allocator for WASM linear memory
// JS heap starts AFTER the WASM stack (which starts at 1MB and grows down)
// and data sections. Use 17MB offset to avoid collision with stack/globals.
let heapBase = 17 * 1024 * 1024;
const align8 = (n) => (n + 7) & ~7;

function malloc(size) {{
  size = Number(size);
  const ptr = heapBase;
  heapBase = align8(heapBase + size);
  // Grow memory if needed (WASM memory starts small, JS heap is at 17MB+)
  if (memory && heapBase > memory.buffer.byteLength) {{
    const needed = Math.ceil((heapBase - memory.buffer.byteLength) / 65536);
    try {{ memory.grow(needed); }} catch(e) {{}}
  }}
  return ptr;
}}

function free(_ptr) {{
  // No-op bump allocator (Phase 1)
}}

// Read a Rayzor string from WASM memory.
// HaxeString struct: {{ ptr: u32, len: u32, cap: u32 }} (12 bytes)
function readString(ptr) {{
  if (!ptr || !memory) return '';
  const view = new DataView(memory.buffer);
  const dataPtr = view.getUint32(ptr, true);     // pointer to UTF-8 bytes
  const len = view.getUint32(ptr + 4, true);     // byte length
  if (len <= 0 || len > 1000000 || dataPtr + len > memory.buffer.byteLength) return '';
  const bytes = new Uint8Array(memory.buffer, dataPtr, len);
  return new TextDecoder().decode(bytes);
}}

// Read a Rayzor Bytes object: [len:i32][data ptr:i32] → Uint8Array
function readBytes(ptr) {{
  if (!memory || !ptr) return new Uint8Array(0);
  const view = new DataView(memory.buffer);
  const len = view.getInt32(ptr, true);
  const dataPtr = view.getInt32(ptr + 4, true);
  return new Uint8Array(memory.buffer, dataPtr, len);
}}

// Pre-call async exports (e.g. GPU device creation) during init.
// Detects Promise-returning functions by actually calling them, but ONLY
// functions whose names suggest device/context creation (to avoid side effects).
const __asyncCache = new Map();
async function _precall_async(hostModule) {{
  for (const [name, fn] of Object.entries(hostModule)) {{
    if (typeof fn !== 'function' || fn.length !== 0) continue;
    // Only try functions that look like device/context creation (not pipeline_begin, cmd_create, etc.)
    if (!name.includes('device_create') && !name.includes('compute_create')) continue;
    try {{
      const result = fn();
      if (result && typeof result.then === 'function') {{
        const resolved = await result;
        __asyncCache.set(name, resolved);
        console.log('[rayzor] pre-resolved: ' + name + ' -> ' + resolved);
      }}
    }} catch(e) {{}}
  }}
}}

// Wrap a Rayzor closure pointer as a callable JS function.
// Closures in WASM linear memory: {{ fn_table_idx: i32, env_ptr: i32 }}
// Reads the function table index and env pointer, returns a JS function
// that calls the WASM function with the env pointer as argument.
let _wasmInstance = null; // set after instantiation
function _wrapFnPtr(closurePtr) {{
  if (!_wasmInstance || !memory || !closurePtr) return () => 0;
  const view = new DataView(memory.buffer);
  const fnIdx = view.getUint32(closurePtr, true);
  // Closure layout: [fn_idx: i32, padding: i32, captures...]
  // Try indirect function table first (if closures have table entries).
  // Fall back to calling via a trampoline export.
  const table = _wasmInstance.exports.__indirect_function_table;
  if (table && fnIdx < table.length) {{
    const fn = table.get(fnIdx);
    if (fn) {{
      // Closure env pointer is at closurePtr + 8 (after fn_idx and padding).
      // The closure function expects (env_ptr: i32) as its first parameter.
      const envPtr = closurePtr + 8;
      return () => fn(envPtr);
    }}
  }}
  return () => 0;
}}

// Build a host adapter that wraps wasm-bindgen exports.
// Handles four cases:
// 1. Async functions → return pre-cached result
// 2. String/byte params → read from Rayzor memory before calling
// 3. Function params → wrap i32 table index as JS function
// 4. Pure i32 functions → pass through directly
function _make_host_adapter(hostModule) {{
  const adapter = {{}};
  for (const [name, fn] of Object.entries(hostModule)) {{
    if (typeof fn !== 'function') {{ adapter[name] = fn; continue; }}

    // If we have a pre-resolved async result, return it as a sync function
    if (__asyncCache.has(name)) {{
      const cached = __asyncCache.get(name);
      adapter[name] = () => cached;
      continue;
    }}

    const src = fn.toString();
    const hasStr = src.includes('passStringToWasm0');
    const hasBytes = src.includes('passArray8ToWasm0');
    // Check param names for callback conventions
    const paramNames = src.match(/^(?:async\s+)?function\s*\w*\(([^)]*)\)/)?.[1]?.split(',').map(s => s.trim()) ?? [];
    const hasFnByName = paramNames.some(p => ['callback','cb','handler','func','f','on_frame'].includes(p.toLowerCase()));
    if (!hasStr && !hasBytes && !hasFnByName) {{
      // Pure i32 function — pass through directly
      adapter[name] = fn;
      continue;
    }}
    // Build a wrapper that converts Rayzor pointers to JS values.
    const strParams = new Set();
    const byteParams = new Set();
    for (const m of src.matchAll(/passStringToWasm0\((\w+),/g)) {{
      const idx = paramNames.indexOf(m[1]);
      if (idx >= 0) strParams.add(idx);
    }}
    for (const m of src.matchAll(/passArray8ToWasm0\((\w+),/g)) {{
      const idx = paramNames.indexOf(m[1]);
      if (idx >= 0) byteParams.add(idx);
    }}
    // Detect function/callback params by name convention.
    // wasm-bindgen passes js_sys::Function via externref — no detectable pattern in JS source.
    // Match common callback param names.
    const fnParams = new Set();
    for (let i = 0; i < paramNames.length; i++) {{
      const pname = paramNames[i].toLowerCase();
      if (pname === 'callback' || pname === 'cb' || pname === 'handler' || pname === 'func' || pname === 'f' || pname === 'on_frame') {{
        fnParams.add(i);
      }}
    }}
    adapter[name] = (...args) => {{
      const converted = args.map((a, i) => {{
        if (strParams.has(i) && typeof a === 'number' && memory) {{
          try {{ return readString(a); }} catch {{ return ''; }}
        }}
        if (byteParams.has(i) && typeof a === 'number' && memory) {{
          try {{ return readBytes(a); }} catch {{ return new Uint8Array(0); }}
        }}
        if (fnParams.has(i) && typeof a === 'number') {{
          return _wrapFnPtr(a);
        }}
        return a;
      }});
      return fn(...converted);
    }};
  }}
  // Override run_loop to use JS-side requestAnimationFrame loop.
  // The window crate's Rust run_loop uses cross-module Closure which doesn't survive.
  if (adapter['rayzor_window_run_loop']) {{
    adapter['rayzor_window_run_loop'] = (winH, cbPtr) => {{
      const cb = _wrapFnPtr(cbPtr);
      // Debug closure struct
      const view = new DataView(memory.buffer);
      const fnIdx = view.getUint32(cbPtr, true);
      const envPtr = view.getUint32(cbPtr + 4, true);
      console.log('[rayzor] runLoop: winH=' + winH + ', closurePtr=' + cbPtr + ', fnIdx=' + fnIdx + ', envPtr=' + envPtr);
      if (!cb) return;
      let frameCount = 0;
      function frame() {{
        try {{
          const cont = cb();
          frameCount++;
          if (frameCount <= 3 || frameCount % 60 === 0) console.log('[rayzor] frame ' + frameCount + ', cont=' + cont);
          if (cont) requestAnimationFrame(frame);
        }} catch(e) {{
          console.error('[rayzor] render loop error:', e);
        }}
      }}
      requestAnimationFrame(frame);
    }};
  }}
  return new Proxy(adapter, {{ get: (t, p) => t[p] ?? ((...a) => 0) }});
}}

// Write a string into WASM memory as HaxeString struct, return struct pointer
function writeString(str) {{
  const encoded = new TextEncoder().encode(str);
  // 1. Write UTF-8 bytes + NUL
  const dataPtr = malloc(encoded.length + 1);
  new Uint8Array(memory.buffer).set(encoded, dataPtr);
  new Uint8Array(memory.buffer)[dataPtr + encoded.length] = 0; // NUL
  // 2. Write HaxeString struct: {{ ptr, len, cap }}
  const structPtr = malloc(12);
  const view = new DataView(memory.buffer);
  view.setUint32(structPtr, dataPtr, true);     // ptr to bytes
  view.setUint32(structPtr + 4, encoded.length, true); // len
  view.setUint32(structPtr + 8, encoded.length, true); // cap
  return structPtr;
}}

// Rayzor runtime imports
const rayzor = {{
  malloc: (size) => malloc(size),
  free: (ptr) => free(ptr),
  haxe_alloc: (size) => malloc(Number(size)),

  // trace() — print to console
  trace: (ptr) => {{
    try {{
      const str = readString(ptr);
      console.log(str);
    }} catch (e) {{
      console.log("[trace: ptr=" + ptr + "]");
    }}
  }},

  // haxe_trace_string_struct — MIR wrapper for trace (takes string pointer)
  haxe_trace_string_struct: (ptr) => {{
    try {{
      const str = readString(ptr);
      console.log(str);
    }} catch (e) {{
      console.log("[trace: ptr=" + ptr + "]");
    }}
  }},

  // String operations
  haxe_string_alloc: (len) => writeString("\0".repeat(Number(len))),
  haxe_string_concat: (a, b) => {{
    try {{
      return writeString(readString(a) + readString(b));
    }} catch {{ return 0; }}
  }},
  haxe_string_length: (ptr) => {{
    try {{ return readString(ptr).length; }} catch {{ return 0; }}
  }},
  haxe_string_from_int: (n) => writeString(String(n)),
  haxe_string_from_float: (f) => writeString(String(f)),
  haxe_int_to_string: (n) => writeString(String(n)),
  haxe_float_to_string: (f) => writeString(String(f)),
  haxe_dynamic_to_string: (ptr) => {{
    if (!ptr) return writeString("null");
    try {{ return writeString(readString(ptr)); }} catch {{ return writeString(String(ptr)); }}
  }},
  haxe_coerce_dynamic_to_int: (ptr) => {{
    if (!ptr) return 0;
    return Number(ptr);
  }},
  haxe_box_int: (n) => n,
  haxe_box_float: (f) => f,
  haxe_box_bool: (b) => b,
  haxe_box_int_ptr: (n) => n,
  haxe_box_float_ptr: (f) => f,
  haxe_unbox_int: (n) => n,
  haxe_unbox_float: (f) => f,
  haxe_unbox_reference_ptr: (ptr) => ptr,
  haxe_concat_string_int: (s, n) => {{
    try {{ return writeString(readString(s) + String(n)); }} catch {{ return 0; }}
  }},
  haxe_concat_string_float: (s, f) => {{
    try {{ return writeString(readString(s) + String(f)); }} catch {{ return 0; }}
  }},
  haxe_concat_string_dynamic: (s, d) => {{
    try {{
      const left = readString(s);
      let right = '';
      if (d) try {{ right = readString(d); }} catch {{ right = String(d); }}
      else right = 'null';
      return writeString(left + right);
    }} catch {{ return 0; }}
  }},

  // Array stub
  haxe_array_get_i64: (_arr, _idx) => BigInt(0),

  // Math (passthrough to JS Math)
  haxe_math_sqrt: (x) => Math.sqrt(x),
  haxe_math_sin: (x) => Math.sin(x),
  haxe_math_cos: (x) => Math.cos(x),
  haxe_math_floor: (x) => Math.floor(x),
  haxe_math_ceil: (x) => Math.ceil(x),
  haxe_math_abs: (x) => Math.abs(x),
  haxe_math_random: () => Math.random(),

  // Thread/sync runtime — overridden by RayzorThreadRuntime.buildImports() in
  // the browser init path below. These fallback stubs apply when the runtime
  // isn't available (e.g. Node.js without shared memory). They return RAW
  // primitives to match the builtin stdlib mapping's expected signatures —
  // matching the wasmtime runner contract.
  rayzor_thread_spawn: (_fnIdx, _envPtr) => 0,
  rayzor_thread_join: (_tid) => 0,
  rayzor_thread_is_finished: (_tid) => 1,
  rayzor_thread_yield_now: () => {{}},
  rayzor_thread_sleep: (_ms) => {{}},
  rayzor_thread_current_id: () => 0,

  // Mutex fallback — simple single-threaded implementation using a Map since
  // Node has no shared memory. Returns RAW primitives (the builtin mapping
  // declares these as returning i32/bool, not boxed DynamicValue*).
  _mutexHandles: new Map(),
  _nextMutexId: 1,
  rayzor_mutex_init: (_val) => {{
    const id = rayzor._nextMutexId++;
    rayzor._mutexHandles.set(id, {{ locked: false }});
    return id;
  }},
  rayzor_mutex_lock: (id) => {{
    const m = rayzor._mutexHandles.get(id);
    if (m) m.locked = true;
    return id;
  }},
  rayzor_mutex_try_lock: (id) => {{
    const m = rayzor._mutexHandles.get(id);
    if (!m) return 0;
    if (m.locked) return 0;
    m.locked = true;
    return 1;
  }},
  rayzor_mutex_is_locked: (id) => {{
    const m = rayzor._mutexHandles.get(id);
    return m && m.locked ? 1 : 0;
  }},
  rayzor_mutex_guard_get: (id) => id,
  rayzor_mutex_unlock: (id) => {{
    const m = rayzor._mutexHandles.get(id);
    if (m) m.locked = false;
  }},
  // Box a value as DynamicValue: {{type_id: i32, value_ptr: i32}}
  // type_id: 0=Int, 1=Float, 2=Bool, 3=String, 4=Object
  _boxBool: function(val) {{
    if (!memory) return 0;
    // Allocate value cell (4 bytes for the bool i32)
    const valPtr = malloc(4);
    new DataView(memory.buffer).setInt32(valPtr, val ? 1 : 0, true);
    // Allocate DynamicValue struct (8 bytes: type_id + value_ptr)
    const dvPtr = malloc(8);
    const v = new DataView(memory.buffer);
    v.setInt32(dvPtr, 2, true);     // type_id = 2 (Bool)
    v.setInt32(dvPtr + 4, valPtr, true); // value_ptr
    return dvPtr;
  }},
  _boxInt: function(val) {{
    if (!memory) return 0;
    const valPtr = malloc(8);
    new DataView(memory.buffer).setBigInt64(valPtr, BigInt(val), true);
    const dvPtr = malloc(8);
    const v = new DataView(memory.buffer);
    v.setInt32(dvPtr, 0, true);     // type_id = 0 (Int)
    v.setInt32(dvPtr + 4, valPtr, true);
    return dvPtr;
  }},
  lock: (id) => {{ if (!memory) return 0; new DataView(memory.buffer).setInt32(id, 1, true); return rayzor._boxInt(id); }},
  try_lock: (id) => {{ if (!memory) return 0; const v = new DataView(memory.buffer); const ok = v.getInt32(id, true) === 0; if (ok) v.setInt32(id, 1, true); return rayzor._boxBool(ok); }},
  is_locked: (id) => {{ if (!memory) return 0; const val = new DataView(memory.buffer).getInt32(id, true) !== 0; return rayzor._boxBool(val); }},
  unlock: (id) => {{ if (!memory) return; new DataView(memory.buffer).setInt32(id, 0, true); }},
  init: (val) => {{ if (!memory) return 0; const p = heapBase; heapBase = align8(heapBase + 8); new DataView(memory.buffer).setInt32(p, 0, true); return p; }},

  // Semaphore
  rayzor_semaphore_init: (n) => {{ if (!memory) return 0; const p = heapBase; heapBase += 4; new Int32Array(memory.buffer)[p>>2] = n; return p; }},
  rayzor_semaphore_acquire: (id) => {{ if (!memory) return; const v = new Int32Array(memory.buffer); const idx = id>>2; while (true) {{ const c = Atomics.load(v,idx); if (c > 0 && Atomics.compareExchange(v,idx,c,c-1) === c) return; }} }},
  rayzor_semaphore_try_acquire: (id) => {{ if (!memory) return 0; const v = new Int32Array(memory.buffer); const idx = id>>2; const c = Atomics.load(v,idx); return (c > 0 && Atomics.compareExchange(v,idx,c,c-1) === c) ? 1 : 0; }},
  sys_semaphore_try_acquire_nowait: (id) => rayzor.rayzor_semaphore_try_acquire(id),

  // Channel — simple ring buffer in linear memory
  rayzor_channel_init: () => {{ if (!memory) return 0; const p = heapBase; heapBase = align8(heapBase + 5*4 + 64*8); const v = new Int32Array(memory.buffer); const b = p>>2; v[b]=0; v[b+1]=0; v[b+2]=0; v[b+3]=64; v[b+4]=0; return p; }},
  rayzor_channel_send: (id, val) => {{ if (!memory) return; const v = new Int32Array(memory.buffer); const dv = new DataView(memory.buffer); const b = id>>2; const cap = v[b+3]; const tail = v[b+1]; dv.setBigUint64(id+20+(tail%cap)*8, BigInt(val), true); v[b+1] = (tail+1)%cap; }},
  rayzor_channel_try_send: (id, val) => {{ rayzor.rayzor_channel_send(id, val); return 1; }},
  rayzor_channel_receive: (id) => {{ if (!memory) return 0; const v = new Int32Array(memory.buffer); const dv = new DataView(memory.buffer); const b = id>>2; const cap = v[b+3]; const head = v[b]; if (head === v[b+1]) return 0; const val = Number(dv.getBigUint64(id+20+(head%cap)*8, true)); v[b] = (head+1)%cap; return val; }},
  rayzor_channel_try_receive: (id) => rayzor.rayzor_channel_receive(id),
  rayzor_channel_close: (id) => {{ if (memory) new Int32Array(memory.buffer)[(id>>2)+2] = 1; }},
  rayzor_channel_is_closed: (id) => {{ if (!memory) return 0; return new Int32Array(memory.buffer)[(id>>2)+2] ? 1 : 0; }},
  rayzor_channel_len: (id) => {{ if (!memory) return 0; const v = new Int32Array(memory.buffer); const b = id>>2; return (v[b+1] - v[b] + v[b+3]) % v[b+3]; }},
  rayzor_channel_capacity: (id) => {{ if (!memory) return 0; return new Int32Array(memory.buffer)[(id>>2)+3]; }},
  rayzor_channel_is_empty: (id) => rayzor.rayzor_channel_len(id) === 0 ? 1 : 0,
  rayzor_channel_is_full: (id) => rayzor.rayzor_channel_len(id) >= rayzor.rayzor_channel_capacity(id) - 1 ? 1 : 0,

  // Future — spawn thread + track result
  rayzor_future_create: (fn, env) => rayzor.rayzor_thread_spawn(fn, env),
  rayzor_future_await: (_id) => 0,
  rayzor_future_then: (_id, _fn, _env) => {{}},
  rayzor_future_poll: (_id) => rayzor.rayzor_thread_is_finished(_id),
  rayzor_future_is_ready: (_id) => rayzor.rayzor_thread_is_finished(_id),
  rayzor_future_all: () => 0,
  rayzor_future_await_timeout: () => 0,
  rayzor_future_race: () => 0,
  rayzor_future_cancel: () => {{}},
  rayzor_future_is_cancelled: () => 0,

  // Arc/Box — identity on WASM (single shared heap)
  rayzor_arc_init: (v) => v,
  rayzor_arc_clone: (v) => v,
  rayzor_arc_get: (v) => v,
  rayzor_arc_strong_count: () => 1,
  rayzor_arc_try_unwrap: (v) => v,
  rayzor_arc_as_ptr: (v) => v,
  rayzor_box_init: (v) => v,
  rayzor_box_unbox: (v) => v,
  rayzor_box_raw: (v) => v,
  rayzor_box_free: () => {{}},

  // EReg — browser RegExp
  // Handle table for regex objects
  _reHandles: new Map(),
  _reNext: 1,
  _reAlloc: function(obj) {{ const h = this._reNext++; this._reHandles.set(h, obj); return h; }},
  _reGet: function(h) {{ return this._reHandles.get(h); }},
  haxe_ereg_new: (patternPtr, flagsPtr) => {{
    try {{
      const pattern = readString(patternPtr);
      const flags = flagsPtr ? readString(flagsPtr) : '';
      const re = new RegExp(pattern, flags.replace(/s/g,''));
      return rayzor._reAlloc({{ re, lastMatch: null, input: '' }});
    }} catch {{ return 0; }}
  }},
  haxe_ereg_match: (h, strPtr) => {{
    const obj = rayzor._reGet(h); if (!obj) return 0;
    const str = readString(strPtr);
    obj.input = str;
    obj.lastMatch = obj.re.exec(str);
    return obj.lastMatch ? 1 : 0;
  }},
  haxe_ereg_matched: (h, n) => {{
    const obj = rayzor._reGet(h); if (!obj?.lastMatch) return 0;
    const s = obj.lastMatch[n]; return s != null ? writeString(s) : 0;
  }},
  haxe_ereg_matched_left: (h) => {{
    const obj = rayzor._reGet(h); if (!obj?.lastMatch) return 0;
    return writeString(obj.input.substring(0, obj.lastMatch.index));
  }},
  haxe_ereg_matched_right: (h) => {{
    const obj = rayzor._reGet(h); if (!obj?.lastMatch) return 0;
    const end = obj.lastMatch.index + obj.lastMatch[0].length;
    return writeString(obj.input.substring(end));
  }},
  haxe_ereg_matched_pos: (h) => {{
    const obj = rayzor._reGet(h); if (!obj?.lastMatch) return 0;
    if (!memory) return 0;
    const p = malloc(8);
    const v = new DataView(memory.buffer);
    v.setInt32(p, obj.lastMatch.index, true);
    v.setInt32(p + 4, obj.lastMatch[0].length, true);
    return p;
  }},
  haxe_ereg_matched_pos_anon: (h) => rayzor.haxe_ereg_matched_pos(h),
  haxe_ereg_match_sub: (h, strPtr, pos, len) => {{
    const obj = rayzor._reGet(h); if (!obj) return 0;
    const str = readString(strPtr).substring(pos, pos + len);
    obj.input = str;
    obj.lastMatch = obj.re.exec(str);
    return obj.lastMatch ? 1 : 0;
  }},
  haxe_ereg_split: (h, strPtr) => {{
    const obj = rayzor._reGet(h); if (!obj) return 0;
    const parts = readString(strPtr).split(obj.re);
    // Return as Haxe Array (simplified — returns first part for now)
    return writeString(parts.join(','));
  }},
  haxe_ereg_replace: (h, strPtr, withPtr) => {{
    const obj = rayzor._reGet(h); if (!obj) return 0;
    const s = readString(strPtr);
    const replacement = readString(withPtr);
    return writeString(s.replace(obj.re, replacement));
  }},
  haxe_ereg_map: (_h, _fn) => 0,
  haxe_ereg_escape: (strPtr) => {{
    const s = readString(strPtr);
    return writeString(s.replace(/[.*+?^${{}}()|[\]\\]/g, '\\$&'));
  }},

  // Compress — browser CompressionStream/DecompressionStream
  rayzor_compress_new: () => 0,
  rayzor_compress_execute: () => 0,
  rayzor_compress_set_flush: () => {{}},
  rayzor_compress_close: () => {{}},
  rayzor_compress_run: () => 0,
  rayzor_uncompress_new: () => 0,
  rayzor_uncompress_execute: () => 0,
  rayzor_uncompress_set_flush: () => {{}},
  rayzor_uncompress_close: () => {{}},

  // Tensor — uses rayzor-gpu wgpu compute backend when available, CPU fallback otherwise.
  // The wgpu host module (gpu/pkg/rayzor_gpu.js) provides synchronous compute via
  // device.poll(Maintain::Wait) inside the wasm-host crate. Tensors are GPU buffer
  // handles + shape; sync reads work because wgpu handles them synchronously in WASM.
  _tensorHandles: new Map(), // tensor_id -> {{ gpu_buf, shape, cpu_data? }}
  _tensorNext: 1,
  _gpuDeviceH: 0, // rayzor-gpu compute device handle (set after _gpuInit)
  _gpuMod: null,  // _rayzor_gpu module ref (set after host init)
  _gpuInit: function() {{
    // Called after wasm-bindgen host init. If rayzor-gpu is loaded, create the compute device.
    if (typeof _rayzor_gpu === 'undefined') return;
    rayzor._gpuMod = _rayzor_gpu;
    // compute_create is async — pre-resolved by _precall_async into the adapter.
    // The host adapter exposes it as a sync function returning the cached result.
    if (typeof _rayzor_gpu.rayzor_gpu_compute_create === 'function') {{
      const result = _rayzor_gpu.rayzor_gpu_compute_create();
      // Result may be a Promise (if not pre-resolved) or a number
      if (typeof result === 'number') {{
        rayzor._gpuDeviceH = result;
      }}
    }}
    if (rayzor._gpuDeviceH > 0) {{
      console.log('[rayzor] Tensor: using wgpu compute backend (device handle ' + rayzor._gpuDeviceH + ')');
    }}
  }},
  _tAlloc: function(data, shape) {{
    // CPU path: store Float64Array directly
    const h = rayzor._tensorNext++;
    rayzor._tensorHandles.set(h, {{ data, shape }});
    return h;
  }},
  // GPU path: allocate GPU buffer, upload data, return tensor handle wrapping buf_h + shape.
  _tAllocGpu: function(data, shape) {{
    const dev = rayzor._gpuDeviceH;
    const numel = data.length;
    const buf_h = rayzor._gpuMod.rayzor_gpu_compute_alloc_buffer(dev, numel, 1); // dtype 1 = F32
    if (!buf_h) return 0;
    // Upload data as f32
    const f32 = new Float32Array(numel);
    for (let i = 0; i < numel; i++) f32[i] = data[i];
    rayzor._gpuMod.rayzor_gpu_compute_buffer_write_f32(dev, buf_h, f32);
    const h = rayzor._tensorNext++;
    rayzor._tensorHandles.set(h, {{ buf_h, shape, gpu: true }});
    return h;
  }},
  _tGet: function(h) {{ return rayzor._tensorHandles.get(rayzor._unboxInt(h)); }},
  // Get numel from a tensor (works for both CPU and GPU paths)
  _tNumel: function(t) {{
    if (!t) return 0;
    if (t.gpu) return t.shape.reduce((a, b) => a * b, 1);
    return t.data.length;
  }},
  // Read a single element from a GPU tensor by flat index
  _tReadGpu: function(t, idx) {{
    if (!t || !t.gpu) return 0;
    return rayzor._gpuMod.rayzor_gpu_compute_buffer_read_f32(rayzor._gpuDeviceH, t.buf_h, idx);
  }},
  // Use GPU when device is available
  _useGpu: function() {{ return rayzor._gpuDeviceH > 0 && rayzor._gpuMod; }},
  // Read a HaxeArray of i32 (shape) — 32-byte header at MIR layout (8-byte stride)
  _readShapeI32: function(arrPtr) {{
    arrPtr = rayzor._unboxInt(arrPtr);
    if (!memory || !arrPtr) return [];
    const v = new DataView(memory.buffer);
    const dataPtr = v.getUint32(arrPtr, true);
    const len = v.getUint32(arrPtr + 8, true);
    const elemSize = v.getUint32(arrPtr + 24, true) || 8;
    const out = [];
    for (let i = 0; i < len; i++) {{
      out.push(v.getInt32(dataPtr + i * elemSize, true));
    }}
    return out;
  }},
  // Read a HaxeArray of f64 (data) from WASM memory
  _readDataF64: function(arrPtr) {{
    arrPtr = rayzor._unboxInt(arrPtr);
    if (!memory || !arrPtr) return [];
    const v = new DataView(memory.buffer);
    const dataPtr = v.getUint32(arrPtr, true);
    const len = v.getUint32(arrPtr + 8, true);
    const elemSize = v.getUint32(arrPtr + 24, true) || 8;
    const out = new Float64Array(len);
    for (let i = 0; i < len; i++) {{
      if (elemSize >= 8) {{
        out[i] = v.getFloat64(dataPtr + i * elemSize, true);
      }} else {{
        out[i] = v.getFloat32(dataPtr + i * elemSize, true);
      }}
    }}
    return out;
  }},
  rayzor_tensor_zeros: (shapePtr, dtype) => {{
    const sh = rayzor._readShapeI32(shapePtr);
    const numel = sh.reduce((a, b) => a * b, 1);
    const data = new Float64Array(numel);
    return rayzor._useGpu() ? rayzor._tAllocGpu(data, sh) : rayzor._tAlloc(data, sh);
  }},
  rayzor_tensor_ones: (shapePtr, dtype) => {{
    const sh = rayzor._readShapeI32(shapePtr);
    const numel = sh.reduce((a, b) => a * b, 1);
    const data = new Float64Array(numel); data.fill(1);
    return rayzor._useGpu() ? rayzor._tAllocGpu(data, sh) : rayzor._tAlloc(data, sh);
  }},
  rayzor_tensor_rand: (shapePtr, dtype) => {{
    const sh = rayzor._readShapeI32(shapePtr);
    const numel = sh.reduce((a, b) => a * b, 1);
    const data = new Float64Array(numel);
    for (let i = 0; i < numel; i++) data[i] = Math.random();
    return rayzor._useGpu() ? rayzor._tAllocGpu(data, sh) : rayzor._tAlloc(data, sh);
  }},
  rayzor_tensor_full: (shapePtr, val, dtype) => {{
    const sh = rayzor._readShapeI32(shapePtr);
    const numel = sh.reduce((a, b) => a * b, 1);
    const data = new Float64Array(numel); data.fill(val);
    return rayzor._useGpu() ? rayzor._tAllocGpu(data, sh) : rayzor._tAlloc(data, sh);
  }},
  rayzor_tensor_from_array: (dataArrPtr, dtype) => {{
    const data = rayzor._readDataF64(dataArrPtr);
    return rayzor._useGpu() ? rayzor._tAllocGpu(data, [data.length]) : rayzor._tAlloc(data, [data.length]);
  }},
  rayzor_tensor_shape: (h) => 0, // returns Array<Int> — complex, stub for now
  rayzor_tensor_ndim: (h) => {{ const t = rayzor._tGet(h); return t ? t.shape.length : 0; }},
  rayzor_tensor_numel: (h) => {{ return rayzor._tNumel(rayzor._tGet(h)); }},
  rayzor_tensor_dtype: (h) => 0,
  rayzor_tensor_shape_ptr: (h) => 0,
  rayzor_tensor_shape_ndim: (h) => {{ const t = rayzor._tGet(h); return t ? t.shape.length : 0; }},
  rayzor_tensor_get: (h, idxArrPtr) => {{
    const t = rayzor._tGet(h); if (!t) return 0;
    const idx = rayzor._readShapeI32(idxArrPtr);
    let flat = 0, stride = 1;
    for (let i = t.shape.length - 1; i >= 0; i--) {{
      flat += (idx[i] || 0) * stride;
      stride *= t.shape[i];
    }}
    if (t.gpu) return rayzor._tReadGpu(t, flat);
    return t.data[flat] || 0;
  }},
  rayzor_tensor_set: (h, idxArrPtr, val) => {{
    const t = rayzor._tGet(h); if (!t || t.gpu) return; // GPU set requires write_buffer; not yet
    const idx = rayzor._readShapeI32(idxArrPtr);
    let flat = 0, stride = 1;
    for (let i = t.shape.length - 1; i >= 0; i--) {{
      flat += (idx[i] || 0) * stride;
      stride *= t.shape[i];
    }}
    t.data[flat] = val;
  }},
  rayzor_tensor_reshape: (h, shapePtr) => {{
    const t = rayzor._tGet(h); if (!t) return 0;
    const sh = rayzor._readShapeI32(shapePtr);
    return rayzor._tAlloc(new Float64Array(t.data), sh);
  }},
  rayzor_tensor_transpose: (h) => {{
    const t = rayzor._tGet(h); if (!t || t.shape.length !== 2) return 0;
    const [m, n] = t.shape;
    const d = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) d[j * m + i] = t.data[i * n + j];
    return rayzor._tAlloc(d, [n, m]);
  }},
  // Binary elementwise — GPU dispatch when both inputs GPU, else CPU
  rayzor_tensor_add: (a, b) => {{
    const at = rayzor._tGet(a), bt = rayzor._tGet(b); if (!at || !bt) return 0;
    if (at.gpu && bt.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_add(rayzor._gpuDeviceH, at.buf_h, bt.buf_h);
      const h = rayzor._tensorNext++;
      rayzor._tensorHandles.set(h, {{ buf_h, shape: [...at.shape], gpu: true }});
      return h;
    }}
    const d = new Float64Array(at.data.length);
    for (let i = 0; i < at.data.length; i++) d[i] = at.data[i] + bt.data[i];
    return rayzor._tAlloc(d, [...at.shape]);
  }},
  rayzor_tensor_sub: (a, b) => {{
    const at = rayzor._tGet(a), bt = rayzor._tGet(b); if (!at || !bt) return 0;
    if (at.gpu && bt.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_sub(rayzor._gpuDeviceH, at.buf_h, bt.buf_h);
      const h = rayzor._tensorNext++;
      rayzor._tensorHandles.set(h, {{ buf_h, shape: [...at.shape], gpu: true }});
      return h;
    }}
    const d = new Float64Array(at.data.length);
    for (let i = 0; i < at.data.length; i++) d[i] = at.data[i] - bt.data[i];
    return rayzor._tAlloc(d, [...at.shape]);
  }},
  rayzor_tensor_mul: (a, b) => {{
    const at = rayzor._tGet(a), bt = rayzor._tGet(b); if (!at || !bt) return 0;
    if (at.gpu && bt.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_mul(rayzor._gpuDeviceH, at.buf_h, bt.buf_h);
      const h = rayzor._tensorNext++;
      rayzor._tensorHandles.set(h, {{ buf_h, shape: [...at.shape], gpu: true }});
      return h;
    }}
    const d = new Float64Array(at.data.length);
    for (let i = 0; i < at.data.length; i++) d[i] = at.data[i] * bt.data[i];
    return rayzor._tAlloc(d, [...at.shape]);
  }},
  rayzor_tensor_div: (a, b) => {{
    const at = rayzor._tGet(a), bt = rayzor._tGet(b); if (!at || !bt) return 0;
    if (at.gpu && bt.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_div(rayzor._gpuDeviceH, at.buf_h, bt.buf_h);
      const h = rayzor._tensorNext++;
      rayzor._tensorHandles.set(h, {{ buf_h, shape: [...at.shape], gpu: true }});
      return h;
    }}
    const d = new Float64Array(at.data.length);
    for (let i = 0; i < at.data.length; i++) d[i] = bt.data[i] !== 0 ? at.data[i] / bt.data[i] : 0;
    return rayzor._tAlloc(d, [...at.shape]);
  }},
  rayzor_tensor_matmul: (a, b) => {{
    const at = rayzor._tGet(a), bt = rayzor._tGet(b); if (!at || !bt) return 0;
    if (at.gpu && bt.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_matmul(rayzor._gpuDeviceH, at.buf_h, bt.buf_h, at.shape[0], at.shape[1], bt.shape[1]);
      const h = rayzor._tensorNext++;
      rayzor._tensorHandles.set(h, {{ buf_h, shape: [at.shape[0], bt.shape[1]], gpu: true }});
      return h;
    }}
    const m = at.shape[0], k = at.shape[1], n = bt.shape[1];
    const d = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {{
      let s = 0; for (let p = 0; p < k; p++) s += at.data[i * k + p] * bt.data[p * n + j];
      d[i * n + j] = s;
    }}
    return rayzor._tAlloc(d, [m, n]);
  }},
  // Unary ops — GPU dispatch when input GPU, else CPU
  rayzor_tensor_sqrt: (h) => {{
    const t = rayzor._tGet(h); if (!t) return 0;
    if (t.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_sqrt(rayzor._gpuDeviceH, t.buf_h);
      const nh = rayzor._tensorNext++;
      rayzor._tensorHandles.set(nh, {{ buf_h, shape: [...t.shape], gpu: true }});
      return nh;
    }}
    const d = new Float64Array(t.data.length);
    for (let i = 0; i < t.data.length; i++) d[i] = Math.sqrt(t.data[i]);
    return rayzor._tAlloc(d, [...t.shape]);
  }},
  rayzor_tensor_exp: (h) => {{
    const t = rayzor._tGet(h); if (!t) return 0;
    if (t.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_exp(rayzor._gpuDeviceH, t.buf_h);
      const nh = rayzor._tensorNext++;
      rayzor._tensorHandles.set(nh, {{ buf_h, shape: [...t.shape], gpu: true }});
      return nh;
    }}
    const d = new Float64Array(t.data.length);
    for (let i = 0; i < t.data.length; i++) d[i] = Math.exp(t.data[i]);
    return rayzor._tAlloc(d, [...t.shape]);
  }},
  rayzor_tensor_log: (h) => {{
    const t = rayzor._tGet(h); if (!t) return 0;
    if (t.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_log(rayzor._gpuDeviceH, t.buf_h);
      const nh = rayzor._tensorNext++;
      rayzor._tensorHandles.set(nh, {{ buf_h, shape: [...t.shape], gpu: true }});
      return nh;
    }}
    const d = new Float64Array(t.data.length);
    for (let i = 0; i < t.data.length; i++) d[i] = Math.log(t.data[i]);
    return rayzor._tAlloc(d, [...t.shape]);
  }},
  rayzor_tensor_relu: (h) => {{
    const t = rayzor._tGet(h); if (!t) return 0;
    if (t.gpu) {{
      const buf_h = rayzor._gpuMod.rayzor_gpu_compute_relu(rayzor._gpuDeviceH, t.buf_h);
      const nh = rayzor._tensorNext++;
      rayzor._tensorHandles.set(nh, {{ buf_h, shape: [...t.shape], gpu: true }});
      return nh;
    }}
    const d = new Float64Array(t.data.length);
    for (let i = 0; i < t.data.length; i++) d[i] = Math.max(0, t.data[i]);
    return rayzor._tAlloc(d, [...t.shape]);
  }},
  // Reductions — sync read via wgpu device.poll(Wait)
  rayzor_tensor_sum: (h) => {{
    const t = rayzor._tGet(h); if (!t) return 0;
    if (t.gpu) return rayzor._gpuMod.rayzor_gpu_compute_sum(rayzor._gpuDeviceH, t.buf_h);
    let s = 0; for (let i = 0; i < t.data.length; i++) s += t.data[i];
    return s;
  }},
  rayzor_tensor_mean: (h) => {{
    const t = rayzor._tGet(h); if (!t || rayzor._tNumel(t) === 0) return 0;
    if (t.gpu) return rayzor._gpuMod.rayzor_gpu_compute_mean(rayzor._gpuDeviceH, t.buf_h);
    let s = 0; for (let i = 0; i < t.data.length; i++) s += t.data[i];
    return s / t.data.length;
  }},
  rayzor_tensor_dot: (a, b) => {{
    const at = rayzor._tGet(a), bt = rayzor._tGet(b); if (!at || !bt) return 0;
    if (at.gpu && bt.gpu) return rayzor._gpuMod.rayzor_gpu_compute_dot(rayzor._gpuDeviceH, at.buf_h, bt.buf_h);
    let s = 0; const n = Math.min(at.data.length, bt.data.length);
    for (let i = 0; i < n; i++) s += at.data[i] * bt.data[i];
    return s;
  }},
  rayzor_tensor_data: (h) => 0,
  rayzor_tensor_free: (h) => {{
    h = rayzor._unboxInt(h);
    const t = rayzor._tensorHandles.get(h);
    if (t && t.gpu && rayzor._gpuMod) {{
      rayzor._gpuMod.rayzor_gpu_compute_free_buffer(rayzor._gpuDeviceH, t.buf_h);
    }}
    rayzor._tensorHandles.delete(h);
  }},
  // Bare-name aliases for forwarder stubs
  Tensor_zeros: (...a) => rayzor.rayzor_tensor_zeros(...a),
  Tensor_ones: (...a) => rayzor.rayzor_tensor_ones(...a),
  Tensor_full: (...a) => rayzor.rayzor_tensor_full(...a),
  Tensor_fromArray: (...a) => rayzor.rayzor_tensor_from_array(...a),
  Tensor_rand: (...a) => rayzor.rayzor_tensor_rand(...a),

  // Networking — fetch + WebSocket for browser
  rayzor_socket_new: () => 0,
  rayzor_socket_connect: () => 0,
  rayzor_socket_bind: () => 0,
  rayzor_socket_listen: () => 0,
  rayzor_socket_accept: () => 0,
  rayzor_socket_close: () => {{}},
  rayzor_socket_read: () => 0,
  rayzor_socket_write: () => 0,
  rayzor_socket_shutdown: () => {{}},
  rayzor_socket_set_blocking: () => {{}},
  rayzor_socket_set_timeout: () => {{}},
  rayzor_socket_set_fast_send: () => {{}},
  rayzor_socket_wait_for_read: () => 0,
  rayzor_socket_select: () => 0,
  rayzor_socket_peer: () => 0,
  rayzor_host_new: () => 0,
  rayzor_host_get_ip: () => 0,
  rayzor_host_to_string: () => 0,
  rayzor_host_reverse: () => 0,
  rayzor_host_localhost: () => 0,
  rayzor_socket_host_info: () => 0,
  rayzor_socket_get_input: () => 0,
  rayzor_socket_get_output: () => 0,
  rayzor_socket_read_byte: () => 0,
  rayzor_socket_read_bytes: () => 0,
  rayzor_socket_write_byte: () => {{}},
  rayzor_socket_write_bytes: () => 0,
  rayzor_socket_write_string: () => 0,
  rayzor_socket_flush: () => {{}},

  // SSL — stub (browser uses HTTPS natively)
  rayzor_ssl_socket_new: () => 0,
  rayzor_ssl_socket_connect: () => 0,
  rayzor_ssl_socket_handshake: () => 0,
  rayzor_ssl_socket_set_hostname: () => {{}},
  rayzor_ssl_socket_set_ca: () => {{}},
  rayzor_ssl_socket_set_certificate: () => {{}},
  rayzor_ssl_socket_peer_certificate: () => 0,
  rayzor_ssl_socket_read: () => 0,
  rayzor_ssl_socket_write: () => 0,
  rayzor_ssl_socket_close: () => {{}},
  rayzor_ssl_socket_set_blocking: () => {{}},
  rayzor_ssl_socket_set_timeout: () => {{}},
  rayzor_ssl_socket_get_input: () => 0,
  rayzor_ssl_socket_get_output: () => 0,
  rayzor_ssl_socket_shutdown: () => {{}},
  rayzor_ssl_socket_set_fast_send: () => {{}},
  rayzor_ssl_socket_read_byte: () => 0,
  rayzor_ssl_socket_read_bytes: () => 0,
  rayzor_ssl_socket_write_byte: () => {{}},
  rayzor_ssl_socket_write_bytes: () => 0,
  rayzor_ssl_socket_write_string: () => 0,
  rayzor_ssl_socket_flush: () => {{}},
  rayzor_ssl_cert_load_file: () => 0,
  rayzor_ssl_cert_load_path: () => 0,
  rayzor_ssl_cert_from_string: () => 0,
  rayzor_ssl_cert_load_defaults: () => 0,
  rayzor_ssl_cert_common_name: () => 0,
  rayzor_ssl_cert_alt_names: () => 0,
  rayzor_ssl_cert_not_before: () => 0,
  rayzor_ssl_cert_not_after: () => 0,
  rayzor_ssl_cert_subject: () => 0,
  rayzor_ssl_cert_issuer: () => 0,
  rayzor_ssl_cert_next: () => 0,
  rayzor_ssl_cert_add: () => 0,
  rayzor_ssl_cert_add_der: () => 0,
  rayzor_ssl_key_load_file: () => 0,
  rayzor_ssl_key_read_pem: () => 0,
  rayzor_ssl_key_read_der: () => 0,
  rayzor_ssl_digest_make: () => 0,
  rayzor_ssl_digest_sign: () => 0,
  rayzor_ssl_digest_verify: () => 0,

  // Typed vectors — pure WASM linear memory implementation
  _vecHandles: new Map(),
  _vecNext: 1,
  _vecAlloc: function(elemSize) {{ const h = this._vecNext++; this._vecHandles.set(h, {{data: [], elemSize}}); return h; }},
  _vecGet: function(h) {{ return this._vecHandles.get(h); }},
  rayzor_vec_i32_new: () => rayzor._vecAlloc(4),
  rayzor_vec_i32_with_capacity: (_cap) => rayzor._vecAlloc(4),
  rayzor_vec_i32_push: (h, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.push(v); }},
  rayzor_vec_i32_pop: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.pop() ?? 0; }},
  rayzor_vec_i32_get: (h, i) => {{ const vec = rayzor._vecGet(h); return vec?.data[i] ?? 0; }},
  rayzor_vec_i32_set: (h, i, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data[i] = v; }},
  rayzor_vec_i32_len: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.length ?? 0; }},
  rayzor_vec_i32_capacity: (h) => rayzor.rayzor_vec_i32_len(h),
  rayzor_vec_i32_is_empty: (h) => rayzor.rayzor_vec_i32_len(h) === 0 ? 1 : 0,
  rayzor_vec_i32_clear: (h) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.length = 0; }},
  rayzor_vec_i32_first: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data[0] ?? 0; }},
  rayzor_vec_i32_last: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.at(-1) ?? 0; }},
  rayzor_vec_i32_sort: (h) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.sort((a,b) => a-b); }},
  rayzor_vec_i32_sort_by: (h, _fn) => rayzor.rayzor_vec_i32_sort(h),
  rayzor_vec_i64_new: () => rayzor._vecAlloc(8),
  rayzor_vec_i64_push: (h, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.push(v); }},
  rayzor_vec_i64_pop: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.pop() ?? 0; }},
  rayzor_vec_i64_get: (h, i) => {{ const vec = rayzor._vecGet(h); return vec?.data[i] ?? 0; }},
  rayzor_vec_i64_set: (h, i, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data[i] = v; }},
  rayzor_vec_i64_len: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.length ?? 0; }},
  rayzor_vec_i64_is_empty: (h) => rayzor.rayzor_vec_i64_len(h) === 0 ? 1 : 0,
  rayzor_vec_i64_clear: (h) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.length = 0; }},
  rayzor_vec_i64_first: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data[0] ?? 0; }},
  rayzor_vec_i64_last: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.at(-1) ?? 0; }},
  rayzor_vec_f64_new: () => rayzor._vecAlloc(8),
  rayzor_vec_f64_push: (h, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.push(v); }},
  rayzor_vec_f64_pop: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.pop() ?? 0; }},
  rayzor_vec_f64_get: (h, i) => {{ const vec = rayzor._vecGet(h); return vec?.data[i] ?? 0; }},
  rayzor_vec_f64_set: (h, i, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data[i] = v; }},
  rayzor_vec_f64_len: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.length ?? 0; }},
  rayzor_vec_f64_is_empty: (h) => rayzor.rayzor_vec_f64_len(h) === 0 ? 1 : 0,
  rayzor_vec_f64_clear: (h) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.length = 0; }},
  rayzor_vec_f64_first: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data[0] ?? 0; }},
  rayzor_vec_f64_last: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.at(-1) ?? 0; }},
  rayzor_vec_f64_sort: (h) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.sort((a,b) => a-b); }},
  rayzor_vec_f64_sort_by: (h, _fn) => rayzor.rayzor_vec_f64_sort(h),
  rayzor_vec_ptr_new: () => rayzor._vecAlloc(4),
  rayzor_vec_ptr_push: (h, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.push(v); }},
  rayzor_vec_ptr_pop: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.pop() ?? 0; }},
  rayzor_vec_ptr_get: (h, i) => {{ const vec = rayzor._vecGet(h); return vec?.data[i] ?? 0; }},
  rayzor_vec_ptr_set: (h, i, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data[i] = v; }},
  rayzor_vec_ptr_len: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.length ?? 0; }},
  rayzor_vec_ptr_is_empty: (h) => rayzor.rayzor_vec_ptr_len(h) === 0 ? 1 : 0,
  rayzor_vec_ptr_clear: (h) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.length = 0; }},
  rayzor_vec_ptr_first: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data[0] ?? 0; }},
  rayzor_vec_ptr_last: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.at(-1) ?? 0; }},
  rayzor_vec_ptr_sort_by: (h, _fn) => {{}},
  rayzor_vec_bool_new: () => rayzor._vecAlloc(1),
  rayzor_vec_bool_push: (h, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.push(v ? 1 : 0); }},
  rayzor_vec_bool_pop: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.pop() ?? 0; }},
  rayzor_vec_bool_get: (h, i) => {{ const vec = rayzor._vecGet(h); return vec?.data[i] ?? 0; }},
  rayzor_vec_bool_set: (h, i, v) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data[i] = v ? 1 : 0; }},
  rayzor_vec_bool_len: (h) => {{ const vec = rayzor._vecGet(h); return vec?.data.length ?? 0; }},
  rayzor_vec_bool_is_empty: (h) => rayzor.rayzor_vec_bool_len(h) === 0 ? 1 : 0,
  rayzor_vec_bool_clear: (h) => {{ const vec = rayzor._vecGet(h); if (vec) vec.data.length = 0; }},

  // Vtable / type registration — no-op stubs (type info not needed at WASM runtime)
  haxe_vtable_init: () => 0,
  haxe_vtable_set_slot: () => {{}},
  haxe_type_register_constructor: () => {{}},
  haxe_register_interface_impl: () => {{}},
  haxe_coerce_dynamic_to_int: (v) => Number(v) || 0,
  rayzor_tcc_create: () => 0,

  // DynamicValue boxing: allocate {{ type_id: u32, value_ptr: u32 }} in WASM memory.
  // Returns the DynamicValue pointer. Used for getter return values that the
  // MIR wrapper chain expects as boxed DynamicValues.
  _boxInt: function(val) {{
    if (!memory) return val;
    const vp = malloc(4);
    new DataView(memory.buffer).setInt32(vp, val, true);
    const dv = malloc(8);
    const v = new DataView(memory.buffer);
    v.setUint32(dv, 3, true);     // type_id = 3 (Int)
    v.setUint32(dv + 4, vp, true); // value_ptr
    return dv;
  }},
  _boxFloat: function(val) {{
    if (!memory) return val;
    const vp = malloc(8);
    new DataView(memory.buffer).setFloat64(vp, val, true);
    const dv = malloc(8);
    const v = new DataView(memory.buffer);
    v.setUint32(dv, 4, true);     // type_id = 4 (Float)
    v.setUint32(dv + 4, vp, true); // value_ptr
    return dv;
  }},
  // DynamicValue unboxing: read {{ type_id: u32, value_ptr: u32 }} from WASM memory.
  // type_id: 0=Void, 1=Null, 2=Bool, 3=Int, 4=Float, 5=String.
  _unboxInt: function(raw) {{
    try {{
      if (!memory || raw <= 65536 || (raw & 3) !== 0) return raw;
      const memSize = memory.buffer.byteLength;
      if (raw + 8 > memSize) return raw;
      const v = new DataView(memory.buffer);
      const tid = v.getUint32(raw, true);
      const vp = v.getUint32(raw + 4, true);
      if ((tid === 2 || tid === 3) && vp > 0 && vp + 4 <= memSize) return v.getInt32(vp, true);
      return raw;
    }} catch(e) {{ return raw; }}
  }},
  _unboxFloat: function(raw) {{
    try {{
      if (!memory || raw <= 65536 || (raw & 3) !== 0) return raw;
      const memSize = memory.buffer.byteLength;
      if (raw + 8 > memSize) return raw;
      const v = new DataView(memory.buffer);
      const tid = v.getUint32(raw, true);
      const vp = v.getUint32(raw + 4, true);
      if (tid === 4 && vp > 0 && vp + 8 <= memSize) return v.getFloat64(vp, true);
      return raw;
    }} catch(e) {{ return raw; }}
  }},
  // haxe.io.Bytes — handle-table backed byte buffer operations
  // All imports use qualified names (haxe_bytes_*) to avoid collisions with other extern classes.
  _bytesHandles: new Map(),
  _bytesNext: 1,
  _bytesGet: function(h) {{ return rayzor._bytesHandles.get(rayzor._unboxInt(h)); }},
  haxe_bytes_alloc: (size) => {{
    if (!memory) return 0;
    size = Number(size);
    const dataPtr = malloc(size);
    new Uint8Array(memory.buffer).fill(0, dataPtr, dataPtr + size);
    const id = rayzor._bytesNext++;
    rayzor._bytesHandles.set(id, {{ dataPtr, len: size }});
    return id;
  }},
  haxe_bytes_of_string: (ptr) => {{
    try {{
      const s = readString(ptr);
      const enc = new TextEncoder().encode(s);
      const dataPtr = malloc(enc.length);
      new Uint8Array(memory.buffer).set(enc, dataPtr);
      const id = rayzor._bytesNext++;
      rayzor._bytesHandles.set(id, {{ dataPtr, len: enc.length }});
      return id;
    }} catch {{ return 0; }}
  }},
  haxe_bytes_length: (h) => {{ const b = rayzor._bytesGet(h); return b ? b.len : 0; }},
  haxe_bytes_get: (h, pos) => {{ if (!memory) return 0; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); if (!b) return 0; return rayzor._boxInt(new Uint8Array(memory.buffer)[b.dataPtr + pos]); }},
  haxe_bytes_set: (h, pos, val) => {{ if (!memory) return; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); val = rayzor._unboxInt(val); if (b) new Uint8Array(memory.buffer)[b.dataPtr + pos] = val; }},
  haxe_bytes_sub: (h, pos, len) => {{
    if (!memory) return 0;
    const b = rayzor._bytesGet(h);
    if (!b) return 0;
    const dataPtr = malloc(len);
    new Uint8Array(memory.buffer).copyWithin(dataPtr, b.dataPtr + pos, b.dataPtr + pos + len);
    const id = rayzor._bytesNext++;
    rayzor._bytesHandles.set(id, {{ dataPtr, len }});
    return id;
  }},
  haxe_bytes_blit: (h, srcPos, dest, destPos, len) => {{
    if (!memory) return;
    const src = rayzor._bytesGet(h);
    const dst = rayzor._bytesGet(dest);
    srcPos = rayzor._unboxInt(srcPos); destPos = rayzor._unboxInt(destPos); len = rayzor._unboxInt(len);
    if (src && dst) new Uint8Array(memory.buffer).copyWithin(dst.dataPtr + destPos, src.dataPtr + srcPos, src.dataPtr + srcPos + len);
  }},
  haxe_bytes_fill: (h, pos, len, val) => {{ if (!memory) return; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); len = rayzor._unboxInt(len); val = rayzor._unboxInt(val); if (b) new Uint8Array(memory.buffer).fill(val, b.dataPtr + pos, b.dataPtr + pos + len); }},
  haxe_bytes_compare: (h1, h2) => {{
    if (!memory) return 0;
    const a = rayzor._bytesGet(h1);
    const bObj = rayzor._bytesGet(h2);
    if (!a || !bObj) return 0;
    const u = new Uint8Array(memory.buffer);
    const len = Math.min(a.len, bObj.len);
    for (let i = 0; i < len; i++) {{
      const av = u[a.dataPtr + i], bv = u[bObj.dataPtr + i];
      if (av !== bv) return rayzor._boxInt(av < bv ? -1 : 1);
    }}
    return rayzor._boxInt(a.len === bObj.len ? 0 : a.len < bObj.len ? -1 : 1);
  }},
  haxe_bytes_to_string: (h) => {{
    if (!memory) return 0;
    const b = rayzor._bytesGet(h);
    if (!b) return 0;
    return writeString(new TextDecoder().decode(new Uint8Array(memory.buffer, b.dataPtr, b.len)));
  }},
  haxe_bytes_get_int16: (h, pos) => {{ if (!memory) return 0; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); if (!b) return 0; return rayzor._boxInt(new DataView(memory.buffer).getInt16(b.dataPtr + pos, true)); }},
  haxe_bytes_get_int32: (h, pos) => {{ if (!memory) return 0; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); if (!b) return 0; return rayzor._boxInt(new DataView(memory.buffer).getInt32(b.dataPtr + pos, true)); }},
  haxe_bytes_get_int64: (h, pos) => {{ if (!memory) return 0; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); if (!b) return 0; return rayzor._boxInt(Number(new DataView(memory.buffer).getBigInt64(b.dataPtr + pos, true))); }},
  haxe_bytes_get_float: (h, pos) => {{ if (!memory) return 0; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); if (!b) return 0; return rayzor._boxFloat(new DataView(memory.buffer).getFloat32(b.dataPtr + pos, true)); }},
  haxe_bytes_get_double: (h, pos) => {{ if (!memory) return 0; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); if (!b) return 0; return rayzor._boxFloat(new DataView(memory.buffer).getFloat64(b.dataPtr + pos, true)); }},
  haxe_bytes_set_int16: (h, pos, val) => {{ if (!memory) return; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); val = rayzor._unboxInt(val); if (b) new DataView(memory.buffer).setInt16(b.dataPtr + pos, val, true); }},
  haxe_bytes_set_int32: (h, pos, val) => {{ if (!memory) return; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); val = rayzor._unboxInt(val); if (b) new DataView(memory.buffer).setInt32(b.dataPtr + pos, val, true); }},
  haxe_bytes_set_int64: (h, pos, val) => {{ if (!memory) return; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); val = rayzor._unboxInt(val); if (b) new DataView(memory.buffer).setBigInt64(b.dataPtr + pos, BigInt(val), true); }},
  haxe_bytes_set_float: (h, pos, val) => {{ if (!memory) return; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); val = rayzor._unboxFloat(val); if (b) new DataView(memory.buffer).setFloat32(b.dataPtr + pos, val, true); }},
  haxe_bytes_set_double: (h, pos, val) => {{ if (!memory) return; const b = rayzor._bytesGet(h); pos = rayzor._unboxInt(pos); val = rayzor._unboxFloat(val); if (b) new DataView(memory.buffer).setFloat64(b.dataPtr + pos, val, true); }},

  // Bare-name aliases for extern class method stubs.
  // The MIR's stdlib-cache creates dead bare stubs ("get", "set") alongside
  // qualified functions ("haxe_bytes_get"). Both become WASM imports.
  // These aliases route bare imports to the handle-table implementations.
  // TODO: Fix at MIR/stdlib-cache level to stop creating bare stubs entirely.
  alloc: (...a) => rayzor.haxe_bytes_alloc(...a),
  ofString: (...a) => rayzor.haxe_bytes_of_string(...a),
  length: (...a) => rayzor.haxe_bytes_length(...a),
  get: (...a) => rayzor.haxe_bytes_get(...a),
  set: (...a) => rayzor.haxe_bytes_set(...a),
  sub: (...a) => rayzor.haxe_bytes_sub(...a),
  blit: (...a) => rayzor.haxe_bytes_blit(...a),
  fill: (...a) => rayzor.haxe_bytes_fill(...a),
  compare: (...a) => rayzor.haxe_bytes_compare(...a),
  toString: (...a) => rayzor.haxe_bytes_to_string(...a),
  getInt16: (...a) => rayzor.haxe_bytes_get_int16(...a),
  getInt32: (...a) => rayzor.haxe_bytes_get_int32(...a),
  getInt64: (...a) => rayzor.haxe_bytes_get_int64(...a),
  getFloat: (...a) => rayzor.haxe_bytes_get_float(...a),
  getDouble: (...a) => rayzor.haxe_bytes_get_double(...a),
  setInt16: (...a) => rayzor.haxe_bytes_set_int16(...a),
  setInt32: (...a) => rayzor.haxe_bytes_set_int32(...a),
  setInt64: (...a) => rayzor.haxe_bytes_set_int64(...a),
  setFloat: (...a) => rayzor.haxe_bytes_set_float(...a),
  setDouble: (...a) => rayzor.haxe_bytes_set_double(...a),
}};

// Proxy: any missing import returns a no-op function
const rayzorProxy = new Proxy(rayzor, {{
  get(target, prop) {{
    if (prop in target) return target[prop];
    return (...args) => {{
      if (typeof args[0] === "bigint") return BigInt(0);
      return 0;
    }};
  }}
}});

async function run() {{
  let wasmBytes;
  if (isNode) {{
    const fs = await import("fs");
    const path = await import("path");
    const url = await import("url");
    const __filename = url.fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    wasmBytes = fs.readFileSync(path.join(__dirname, "{filename}"));
  }} else {{
    const resp = await fetch("{filename}");
    wasmBytes = await resp.arrayBuffer();
  }}

  // WASI polyfill (minimal — just fd_write for stdout)
  const wasi_snapshot_preview1 = {{
    fd_write: (fd, iovs_ptr, iovs_len, nwritten_ptr) => {{
      if (!memory) return 0;
      try {{
        const mem = new Uint8Array(memory.buffer);
        const view = new DataView(memory.buffer);
        let written = 0;
        for (let i = 0; i < iovs_len; i++) {{
          const base = iovs_ptr + i * 8;
          if (base + 8 > mem.length) break;
          const ptr = view.getUint32(base, true);
          const len = view.getUint32(base + 4, true);
          if (len === 0 || ptr + len > mem.length) continue;
          const bytes = mem.slice(ptr, ptr + len);
          if (fd === 1 || fd === 2) {{
            if (isNode) {{ process.stdout.write(Buffer.from(bytes)); }}
            else {{ console.log(new TextDecoder().decode(bytes)); }}
          }}
          written += len;
        }}
        if (nwritten_ptr + 4 <= mem.length) {{
          view.setUint32(nwritten_ptr, written, true);
        }}
        return 0;
      }} catch(e) {{ return 0; }}
    }},
    environ_get: () => 0,
    environ_sizes_get: (count_ptr, buf_size_ptr) => {{
      if (memory) {{
        const view = new DataView(memory.buffer);
        view.setUint32(count_ptr, 0, true);
        view.setUint32(buf_size_ptr, 0, true);
      }}
      return 0;
    }},
    proc_exit: (code) => {{ if (isNode) process.exit(code); }},
    fd_close: () => 0,
    fd_read: () => 0,
    fd_seek: () => 0,
    fd_fdstat_get: () => 0,
    fd_filestat_get: () => 0,
    fd_prestat_get: () => 8,
    fd_prestat_dir_name: () => 8,
    path_open: () => 44,
    args_get: () => 0,
    args_sizes_get: (argc, buf) => {{
      if (memory) {{
        const v = new DataView(memory.buffer);
        v.setUint32(argc, 0, true);
        v.setUint32(buf, 0, true);
      }}
      return 0;
    }},
    clock_time_get: (id, prec, out) => {{
      if (memory) {{
        const v = new DataView(memory.buffer);
        v.setBigUint64(out, BigInt(Math.round(performance.now() * 1e6)), true);
      }}
      return 0;
    }},
    random_get: (buf, len) => {{
      if (memory) crypto.getRandomValues(new Uint8Array(memory.buffer, buf, len));
      return 0;
    }},
  }};

  // Proxy: catch any missing WASI calls gracefully
  const wasiProxy = new Proxy(wasi_snapshot_preview1, {{
    get: (target, prop) => target[prop] ?? ((...args) => 0),
  }});

  // Initialize wasm-bindgen host modules (loads _bg.wasm, pre-resolves async)
  {host_init}

  // Initialize Tensor compute backend (uses rayzor-gpu host if available)
  if (typeof rayzor._gpuInit === 'function') rayzor._gpuInit();

  // Set up the browser thread runtime (Web Worker pool + Atomics primitives).
  // In Node we keep the inline stubs since Node has no window.Worker API.
  let threadRuntime = null;
  if (!isNode && typeof Worker !== 'undefined') {{
    try {{
      threadRuntime = new RayzorThreadRuntime();
      // Thread<T>.join() expects a boxed DynamicValue* so compiled Haxe code
      // unboxes it through haxe_unbox_int. Pass the harness' `_boxInt` helper
      // into the runtime so join() can box its cached result the same way the
      // native rayzor_thread_join does via haxe_box_int_ptr.
      threadRuntime.setBoxHelpers({{
        boxInt: (v) => rayzor._boxInt(v),
      }});
      const threadImports = threadRuntime.buildImports();
      for (const [k, v] of Object.entries(threadImports)) {{
        rayzor[k] = v;
      }}
    }} catch (e) {{
      console.warn('[rayzor] thread runtime init failed:', e);
    }}
  }}

  const {{ instance }} = await WebAssembly.instantiate(wasmBytes, {{
    rayzor: rayzorProxy,
    wasi_snapshot_preview1: wasiProxy,
{host_import_objects}  }});

  memory = instance.exports.memory;
  _wasmInstance = instance;

  // Lazily boot the Worker pool once we have a WASM module + memory.
  if (threadRuntime) {{
    try {{
      // `await` here is fine because run() is already async.
      const workerUrl = new URL('./rayzor_worker.js', import.meta.url);
      await threadRuntime.init(wasmBytes, memory, instance, workerUrl);
    }} catch (e) {{
      console.warn('[rayzor] thread runtime worker pool init failed:', e);
    }}
  }}

  if (instance.exports._start) {{
    instance.exports._start();
  }} else {{
    console.error("No _start export found. Available exports:", Object.keys(instance.exports));
  }}
}}

run().catch(e => {{ console.error("WASM error:", e); if (typeof process !== "undefined") process.exitCode = 1; }});
"#,
        filename = wasm_filename,
        host_imports = host_imports,
        host_init = host_init,
        host_import_objects = host_import_objects
    )
}
