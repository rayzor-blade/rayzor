//! WASM Bindgen — generates ES6 class wrappers from @:export annotated Haxe classes.
//!
//! Produces JavaScript ES6 modules with class wrappers that call into
//! WASM-exported functions with proper memory management.
//!
//! Given a Haxe class:
//! ```haxe
//! @:export
//! class Vec2 {
//!     public var x:Float;
//!     public var y:Float;
//!     public function new(x:Float, y:Float) { ... }
//!     public function length():Float { ... }
//!     public static function zero():Vec2 { ... }
//! }
//! ```
//!
//! Generates:
//! ```js
//! export class Vec2 {
//!   #ptr;
//!   constructor(x, y) { this.#ptr = __exports.malloc(24); __exports.Vec2_new(this.#ptr, x, y); }
//!   length() { return __exports.Vec2_length(this.#ptr); }
//!   static zero() { return Vec2.__wrap(__exports.Vec2_zero()); }
//!   static __wrap(ptr) { const o = Object.create(Vec2.prototype); o.#ptr = ptr; return o; }
//!   get __pointer() { return this.#ptr; }
//! }
//! ```

use crate::ir::modules::IrModule;
use crate::ir::IrType;
use std::collections::BTreeMap;

/// Metadata for an exported class, collected from MIR.
#[derive(Debug)]
pub struct ExportedClass {
    pub name: String,
    pub alloc_size: u32,
    pub constructor: Option<ExportedMethod>,
    pub instance_methods: Vec<ExportedMethod>,
    pub static_methods: Vec<ExportedMethod>,
}

/// Metadata for an exported method.
#[derive(Debug)]
pub struct ExportedMethod {
    pub name: String,
    /// Parameter names (excluding 'this' for instance methods/constructors)
    pub params: Vec<String>,
    /// Parameter types (excluding 'this'), parallel to params
    pub param_types: Vec<IrType>,
    pub return_type: IrType,
    /// Whether return type is a class pointer that should be wrapped
    pub returns_class: Option<String>,
}

/// Collect @:jsImport mappings from MIR modules.
/// Returns a map of (js_module_name → Vec<(wasm_function_name, js_import_name)>).
pub fn collect_js_imports(modules: &[&IrModule]) -> BTreeMap<String, Vec<(String, String)>> {
    let mut result: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for module in modules {
        for func in module.functions.values() {
            if let Some((ref js_module, ref js_name)) = func.js_import {
                result
                    .entry(js_module.clone())
                    .or_default()
                    .push((func.name.clone(), js_name.clone()));
            }
        }
    }
    result
}

/// Collect exported class metadata from MIR modules.
pub fn collect_exported_classes(
    modules: &[&IrModule],
    class_alloc_sizes: &BTreeMap<String, u64>,
) -> Vec<ExportedClass> {
    // Group exported functions by class name
    let mut class_map: BTreeMap<String, ExportedClass> = BTreeMap::new();

    // Also collect all exported class names so we know which return types to wrap
    let mut exported_class_names: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();

    // First pass: discover all exported class names
    for module in modules {
        for func in module.functions.values() {
            if !func.wasm_export {
                continue;
            }
            if let Some(ref qn) = func.qualified_name {
                if let Some(dot) = qn.rfind('.') {
                    exported_class_names.insert(qn[..dot].to_string());
                }
            }
        }
    }

    // Second pass: collect methods per class
    for module in modules {
        for func in module.functions.values() {
            if !func.wasm_export {
                continue;
            }
            let qn = match func.qualified_name.as_deref() {
                Some(qn) => qn,
                None => continue,
            };
            let dot = match qn.rfind('.') {
                Some(d) => d,
                None => continue,
            };
            let class_name = &qn[..dot];
            let method_name = &qn[dot + 1..];

            let class = class_map.entry(class_name.to_string()).or_insert_with(|| {
                let alloc_size = class_alloc_sizes.get(class_name).copied().unwrap_or(64) as u32;
                ExportedClass {
                    name: class_name.to_string(),
                    alloc_size,
                    constructor: None,
                    instance_methods: Vec::new(),
                    static_methods: Vec::new(),
                }
            });

            let params = &func.signature.parameters;
            let has_this = !params.is_empty() && params[0].name == "this";

            // Determine if return type is an exported class (pointer that should be wrapped)
            let returns_class = match &func.signature.return_type {
                IrType::Ptr(_) => {
                    // Check if qualified_name hints at a class return
                    // For now, we can't easily determine this from MIR alone
                    // The JS wrapper will need to handle this via convention
                    None
                }
                _ => None,
            };

            // Build param list (skip 'this' for instance methods)
            let skip = if has_this { 1 } else { 0 };
            let param_names: Vec<String> = params[skip..]
                .iter()
                .map(|p| {
                    let name = &p.name;
                    if name.starts_with("param_") || name.is_empty() {
                        format!("arg{}", p.reg.as_u32())
                    } else {
                        sanitize_js_name(name)
                    }
                })
                .collect();
            let param_types: Vec<IrType> = params[skip..].iter().map(|p| p.ty.clone()).collect();

            let method = ExportedMethod {
                name: method_name.to_string(),
                params: param_names,
                param_types,
                return_type: func.signature.return_type.clone(),
                returns_class,
            };

            if method_name == "new" {
                class.constructor = Some(method);
            } else if has_this {
                class.instance_methods.push(method);
            } else {
                class.static_methods.push(method);
            }
        }
    }

    class_map.into_values().collect()
}

/// Generate ES6 module with class wrappers for WASM exports.
pub fn generate_es6_bindings(classes: &[ExportedClass], wasm_filename: &str) -> String {
    let mut js = String::new();

    js.push_str("// Auto-generated by rayzor build --target wasm\n");
    js.push_str("// ES6 class wrappers for WASM exports\n\n");

    // Module-level state
    js.push_str("let __exports = null;\n");
    js.push_str("let __memory = null;\n\n");

    // String helpers
    js.push_str(STRING_HELPERS);

    // Generate class definitions
    for class in classes {
        generate_class(&mut js, class);
        js.push('\n');
    }

    // Generate loadRayzor() function
    generate_loader(&mut js, classes, wasm_filename);

    js
}

/// Generate a single ES6 class wrapper.
fn generate_class(js: &mut String, class: &ExportedClass) {
    let cn = &class.name;

    js.push_str(&format!("export class {} {{\n", cn));
    js.push_str("  #ptr;\n\n");

    // Constructor
    if let Some(ctor) = &class.constructor {
        let params_str = ctor.params.join(", ");
        // If called with a single __raw_ptr__ argument, use it directly (for __wrap)
        js.push_str(&format!("  constructor({}) {{\n", params_str));
        js.push_str(&format!(
            "    this.#ptr = __exports.malloc({});\n",
            class.alloc_size
        ));
        let call_args = std::iter::once("this.#ptr".to_string())
            .chain(ctor.params.iter().map(|p| coerce_arg(p)))
            .collect::<Vec<_>>()
            .join(", ");
        js.push_str(&format!("    __exports.{}_new({});\n", cn, call_args));
        js.push_str("  }\n\n");
    }

    // Instance methods
    for method in &class.instance_methods {
        let params_str = method.params.join(", ");
        js.push_str(&format!("  {}({}) {{\n", method.name, params_str));
        let call_args = std::iter::once("this.#ptr".to_string())
            .chain(method.params.iter().map(|p| coerce_arg(p)))
            .collect::<Vec<_>>()
            .join(", ");
        let export_name = format!("{}_{}", cn, method.name);
        if matches!(method.return_type, IrType::Void) {
            js.push_str(&format!("    __exports.{}({});\n", export_name, call_args));
        } else {
            js.push_str(&format!(
                "    return __exports.{}({});\n",
                export_name, call_args
            ));
        }
        js.push_str("  }\n\n");
    }

    // Static methods
    for method in &class.static_methods {
        let params_str = method.params.join(", ");
        js.push_str(&format!("  static {}({}) {{\n", method.name, params_str));
        let call_args = method
            .params
            .iter()
            .map(|p| coerce_arg(p))
            .collect::<Vec<_>>()
            .join(", ");
        let export_name = format!("{}_{}", cn, method.name);
        if matches!(method.return_type, IrType::Void) {
            js.push_str(&format!("    __exports.{}({});\n", export_name, call_args));
        } else {
            js.push_str(&format!(
                "    return __exports.{}({});\n",
                export_name, call_args
            ));
        }
        js.push_str("  }\n\n");
    }

    // __wrap static method — construct from raw pointer
    js.push_str(&format!("  static __wrap(ptr) {{\n"));
    js.push_str(&format!(
        "    const obj = Object.create({}.prototype);\n",
        cn
    ));
    js.push_str("    obj.#ptr = ptr;\n");
    js.push_str("    return obj;\n");
    js.push_str("  }\n\n");

    // Pointer accessor for passing instances to other WASM functions
    js.push_str("  get __pointer() { return this.#ptr; }\n");

    js.push_str("}\n");
}

/// Generate the loadRayzor() async loader function.
fn generate_loader(js: &mut String, classes: &[ExportedClass], wasm_filename: &str) {
    js.push_str(&format!(
        r#"
export async function loadRayzor(wasmPath) {{
  const isNode = typeof process !== "undefined" && process.versions?.node;
  let wasmBytes;
  if (isNode) {{
    const fs = await import("fs");
    wasmBytes = fs.readFileSync(wasmPath || "{}");
  }} else {{
    const resp = await fetch(wasmPath || "{}");
    wasmBytes = await resp.arrayBuffer();
  }}

  const wasi = {{
    fd_write: (fd, iovs, iovs_len, nwritten) => {{
      if (!__memory) return 0;
      const view = new DataView(__memory.buffer);
      let written = 0;
      for (let i = 0; i < iovs_len; i++) {{
        const ptr = view.getUint32(iovs + i * 8, true);
        const len = view.getUint32(iovs + i * 8 + 4, true);
        if (len > 0 && ptr + len <= __memory.buffer.byteLength) {{
          const bytes = new Uint8Array(__memory.buffer, ptr, len);
          if (isNode) process.stdout.write(Buffer.from(bytes));
          else console.log(new TextDecoder().decode(bytes));
          written += len;
        }}
      }}
      if (nwritten + 4 <= __memory.buffer.byteLength)
        view.setUint32(nwritten, written, true);
      return 0;
    }},
    environ_get: () => 0,
    environ_sizes_get: (c, b) => {{
      if (__memory) {{
        const v = new DataView(__memory.buffer);
        v.setUint32(c, 0, true);
        v.setUint32(b, 0, true);
      }}
      return 0;
    }},
    proc_exit: (code) => {{ if (isNode) process.exit(code); }},
    fd_close: () => 0,
    fd_seek: () => 0,
    fd_read: () => 0,
    fd_fdstat_get: () => 0,
    fd_prestat_get: () => {{ return 8; }}, // EBADF
    fd_prestat_dir_name: () => {{ return 8; }},
    path_open: () => {{ return 44; }}, // ENOSYS
    clock_time_get: (id, precision, out) => {{
      if (__memory) {{
        const v = new DataView(__memory.buffer);
        const now = BigInt(Math.round(performance.now() * 1e6));
        v.setBigUint64(out, now, true);
      }}
      return 0;
    }},
    random_get: (buf, len) => {{
      if (__memory) {{
        const bytes = new Uint8Array(__memory.buffer, buf, len);
        crypto.getRandomValues(bytes);
      }}
      return 0;
    }},
    args_get: () => 0,
    args_sizes_get: (argc, argv_buf_size) => {{
      if (__memory) {{
        const v = new DataView(__memory.buffer);
        v.setUint32(argc, 0, true);
        v.setUint32(argv_buf_size, 0, true);
      }}
      return 0;
    }},
  }};

  // Proxy catches any unimplemented WASI calls gracefully
  const wasiProxy = new Proxy(wasi, {{
    get: (target, prop) => target[prop] ?? ((...args) => 0),
  }});

  const {{ instance }} = await WebAssembly.instantiate(wasmBytes, {{
    wasi_snapshot_preview1: wasiProxy,
  }});

  __exports = instance.exports;
  __memory = instance.exports.memory;

  // Call _start to initialize runtime
  if (__exports._start) __exports._start();

  return {{
    _instance: instance,
    _memory: __memory,
    readString,
    writeString,
"#,
        wasm_filename, wasm_filename
    ));

    // Add class references to the return object
    for class in classes {
        js.push_str(&format!("    {},\n", class.name));
    }

    // Add raw export access
    js.push_str("    exports: __exports,\n");

    js.push_str("  };\n}\n");
}

/// Coerce a JS argument — unwrap class instances to their raw pointer.
/// Uses duck typing: if the argument has a `__pointer` property, extract it.
fn coerce_arg(name: &str) -> String {
    format!("{}?.__pointer ?? {}", name, name)
}

/// Sanitize a name for use as a JS identifier.
fn sanitize_js_name(name: &str) -> String {
    // JS reserved words and common conflicts
    match name {
        "new" | "class" | "return" | "var" | "let" | "const" | "function" | "this" | "super"
        | "switch" | "case" | "default" | "break" | "continue" | "for" | "while" | "do" | "if"
        | "else" | "try" | "catch" | "finally" | "throw" | "typeof" | "instanceof" | "delete"
        | "void" | "in" | "of" | "with" | "yield" | "await" | "async" | "import" | "export"
        | "from" => format!("_{}", name),
        _ => name.to_string(),
    }
}

/// Generate TypeScript .d.ts definitions for exported classes.
pub fn generate_typescript_defs(classes: &[ExportedClass]) -> String {
    let mut ts = String::new();
    ts.push_str("// Auto-generated by rayzor build --target wasm\n");
    ts.push_str("// TypeScript definitions for WASM exports\n\n");

    for class in classes {
        ts.push_str(&format!("export declare class {} {{\n", class.name));

        // Constructor
        if let Some(ctor) = &class.constructor {
            let params: Vec<String> = ctor
                .params
                .iter()
                .map(|p| format!("{}: {}", p, ir_type_to_ts(&ctor.return_type)))
                .collect();
            // Constructor params — use number for all for now
            let params: Vec<String> = ctor
                .params
                .iter()
                .map(|p| format!("{}: number", p))
                .collect();
            ts.push_str(&format!("  constructor({});\n", params.join(", ")));
        }

        // Instance methods
        for method in &class.instance_methods {
            let params: Vec<String> = method
                .params
                .iter()
                .map(|p| format!("{}: number", p))
                .collect();
            let ret = ir_type_to_ts(&method.return_type);
            ts.push_str(&format!(
                "  {}({}): {};\n",
                method.name,
                params.join(", "),
                ret
            ));
        }

        // Static methods
        for method in &class.static_methods {
            let params: Vec<String> = method
                .params
                .iter()
                .map(|p| format!("{}: number", p))
                .collect();
            let ret = ir_type_to_ts(&method.return_type);
            ts.push_str(&format!(
                "  static {}({}): {};\n",
                method.name,
                params.join(", "),
                ret
            ));
        }

        ts.push_str(&format!("  static __wrap(ptr: number): {};\n", class.name));
        ts.push_str("  readonly __pointer: number;\n");
        ts.push_str("}\n\n");
    }

    ts.push_str("export interface RayzorModule {\n");
    ts.push_str("  _instance: WebAssembly.Instance;\n");
    ts.push_str("  _memory: WebAssembly.Memory;\n");
    ts.push_str("  readString(ptr: number): string;\n");
    ts.push_str("  writeString(str: string): number;\n");
    for class in classes {
        ts.push_str(&format!("  {}: typeof {};\n", class.name, class.name));
    }
    ts.push_str("  exports: WebAssembly.Exports;\n");
    ts.push_str("}\n\n");
    ts.push_str("export function loadRayzor(wasmPath?: string): Promise<RayzorModule>;\n");
    ts
}

fn ir_type_to_ts(ty: &IrType) -> &'static str {
    match ty {
        IrType::Void => "void",
        IrType::Bool => "boolean",
        IrType::I8
        | IrType::I16
        | IrType::I32
        | IrType::U8
        | IrType::U16
        | IrType::U32
        | IrType::I64
        | IrType::U64 => "number",
        IrType::F32 | IrType::F64 => "number",
        IrType::String => "string",
        IrType::Ptr(_) => "number",
        _ => "any",
    }
}

const STRING_HELPERS: &str = r#"
function readString(ptr) {
  if (!__memory || ptr === 0) return "";
  const view = new DataView(__memory.buffer);
  const dataPtr = view.getUint32(ptr, true);
  const len = view.getUint32(ptr + 4, true);
  if (dataPtr === 0 || len === 0) return "";
  const bytes = new Uint8Array(__memory.buffer, dataPtr, len);
  return new TextDecoder().decode(bytes);
}

function writeString(str) {
  if (!__exports) return 0;
  const encoded = new TextEncoder().encode(str);
  const dataPtr = __exports.malloc?.(encoded.length + 1) ?? 0;
  if (dataPtr === 0) return 0;
  new Uint8Array(__memory.buffer).set(encoded, dataPtr);
  new Uint8Array(__memory.buffer)[dataPtr + encoded.length] = 0;
  const hsPtr = __exports.malloc?.(12) ?? 0;
  if (hsPtr === 0) return 0;
  const view = new DataView(__memory.buffer);
  view.setUint32(hsPtr, dataPtr, true);
  view.setUint32(hsPtr + 4, encoded.length, true);
  view.setUint32(hsPtr + 8, encoded.length, true);
  return hsPtr;
}

"#;
