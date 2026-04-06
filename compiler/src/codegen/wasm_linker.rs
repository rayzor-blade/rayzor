//! WASM linker: merges a user `.wasm` module with a runtime `.wasm` module
//! into a single self-contained `.wasm` binary.
//!
//! The user module imports ~278 functions from the "rayzor" namespace (runtime
//! functions like `haxe_trace_string_struct`, `haxe_array_new`, etc.).
//! The runtime module imports only WASI functions (4 from "wasi_snapshot_preview1"),
//! exports ~59 functions, and has ~167 internal helpers.
//!
//! After linking, the output has:
//! - Only WASI imports remaining
//! - All runtime + user functions merged into a single function index space
//! - A single shared linear memory
//! - Merged globals, data sections, and a unified type section
//!
//! # Merged function index layout
//!
//! ```text
//! [0..N_wasi)                                     = WASI imports
//! [N_wasi..N_wasi+N_rt_internal)                  = runtime internal functions
//! [N_wasi+N_rt_internal..+N_stubs)                = stubs for unresolved user imports
//! [N_wasi+N_rt_internal+N_stubs..+N_user)         = user internal functions
//! ```
//!
//! # Algorithm
//!
//! 1. Parse both modules with `wasmparser::Parser`, extracting types, imports,
//!    functions, exports, memory, globals, and data segments.
//! 2. Build a merged type section (deduplicating identical function signatures).
//! 3. Build index remapping tables for both modules.
//! 4. Re-encode every function body, rewriting `Call`, `RefFunc`, `GlobalGet`,
//!    `GlobalSet`, `ReturnCall`, and `CallIndirect` instructions with merged indices.
//! 5. Emit the merged module via `wasm-encoder`.

use std::borrow::Cow;
use std::collections::BTreeMap;

use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection, Function,
    FunctionSection, GlobalSection, GlobalType as EncGlobalType, ImportSection, Instruction,
    MemArg as EncMemArg, MemorySection, MemoryType as EncMemoryType, Module, TypeSection, ValType,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct WasmLinker;

impl WasmLinker {
    /// Link a user WASM module with the pre-built runtime WASM.
    /// Returns a single self-contained `.wasm` binary with only WASI imports remaining.
    pub fn link(user_wasm: &[u8], runtime_wasm: &[u8]) -> Result<Vec<u8>, String> {
        Self::link_with_hosts(user_wasm, runtime_wasm, &BTreeMap::new())
    }

    /// Link with host module function map.
    /// `host_functions` maps function names (e.g. "rayzor_gpu_gfx_device_create")
    /// to host module names (e.g. "rayzor-gpu"). Matching "rayzor" imports are
    /// preserved in the output as imports from the host module instead of being stubbed.
    pub fn link_with_hosts(
        user_wasm: &[u8],
        runtime_wasm: &[u8],
        host_functions: &BTreeMap<String, String>,
    ) -> Result<Vec<u8>, String> {
        // 1. Parse both modules.
        let rt = ParsedModule::parse(runtime_wasm, "runtime")?;
        let user = ParsedModule::parse(user_wasm, "user")?;

        // 2. Build the merged module.
        let mut linker = LinkerCtx::new();
        linker.host_functions = host_functions.clone();
        linker.merge(&rt, &user)?;
        linker.encode(&rt, &user)
    }

    /// Scan a JS file for `export function <name>(` lines and return the names.
    /// Works with wasm-bindgen output (ES6 module format).
    pub fn scan_js_exports(js_source: &str) -> Vec<String> {
        let mut names = Vec::new();
        for line in js_source.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("export function ") {
                if let Some(paren) = rest.find('(') {
                    let name = rest[..paren].trim();
                    if !name.is_empty() {
                        names.push(name.to_string());
                    }
                }
            }
        }
        names
    }
}

// ---------------------------------------------------------------------------
// Parsed module representation
// ---------------------------------------------------------------------------

/// A function signature: (params, results).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct FuncSig {
    params: Vec<ValType>,
    results: Vec<ValType>,
}

/// An import from the original module.
#[derive(Clone, Debug)]
struct ParsedImport {
    module: String,
    name: String,
    /// Index into the module's local type section.
    type_idx: u32,
}

/// A function defined internally in the module (has code body).
#[derive(Clone, Debug)]
struct ParsedFunction {
    /// Index into the module's local type section.
    type_idx: u32,
}

/// An export from the module.
#[derive(Clone, Debug)]
struct ParsedExport {
    name: String,
    kind: ExportItemKind,
    /// Index in the original module's index space.
    index: u32,
}

#[derive(Clone, Debug, PartialEq)]
enum ExportItemKind {
    Func,
    Memory,
    Global,
    Table,
}

/// A global variable.
#[derive(Clone, Debug)]
struct ParsedGlobal {
    val_type: ValType,
    mutable: bool,
    /// The init expression, decoded as a simple constant.
    init: GlobalInit,
}

#[derive(Clone, Debug)]
enum GlobalInit {
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
    /// GlobalGet referencing another global (used for imported globals).
    GlobalGet(u32),
}

/// An active data segment.
#[derive(Clone, Debug)]
struct ParsedData {
    /// Memory index (usually 0).
    memory_idx: u32,
    /// Offset in linear memory.
    offset: u32,
    /// Raw bytes.
    data: Vec<u8>,
}

/// A fully parsed WASM module.
struct ParsedModule {
    types: Vec<FuncSig>,
    /// Function imports (the first N function indices).
    func_imports: Vec<ParsedImport>,
    /// Internal functions (index starts after imports).
    functions: Vec<ParsedFunction>,
    /// Raw code bodies, one per internal function.
    code_bodies: Vec<Vec<u8>>,
    exports: Vec<ParsedExport>,
    /// Export name -> original function index (in this module's index space).
    export_func_map: BTreeMap<String, u32>,
    memory: Option<(u64, Option<u64>)>,
    globals: Vec<ParsedGlobal>,
    data_segments: Vec<ParsedData>,
    /// Total number of function imports (= first internal func index).
    num_func_imports: u32,
    /// Tables: (initial_size, maximum_size)
    tables: Vec<(u32, Option<u32>)>,
    /// Element segments: (table_idx, offset, func_indices)
    elements: Vec<(u32, u32, Vec<u32>)>,
}

impl ParsedModule {
    fn parse(wasm: &[u8], label: &str) -> Result<Self, String> {
        use wasmparser::{Parser, Payload};

        let mut types = Vec::new();
        let mut func_imports = Vec::new();
        let mut functions = Vec::new();
        let mut code_bodies = Vec::new();
        let mut exports = Vec::new();
        let mut export_func_map = BTreeMap::new();
        let mut memory = None;
        let mut globals = Vec::new();
        let mut tables = Vec::new();
        let mut elements = Vec::new();
        let mut data_segments = Vec::new();
        let mut num_func_imports = 0u32;

        let parser = Parser::new(0);
        for payload in parser.parse_all(wasm) {
            let payload = payload.map_err(|e| format!("{label}: parse error: {e}"))?;
            match payload {
                Payload::TypeSection(reader) => {
                    for rec_group in reader {
                        let rec_group =
                            rec_group.map_err(|e| format!("{label}: type error: {e}"))?;
                        for sub_type in rec_group.into_types() {
                            if let wasmparser::CompositeInnerType::Func(func_type) =
                                sub_type.composite_type.inner
                            {
                                types.push(FuncSig {
                                    params: func_type
                                        .params()
                                        .iter()
                                        .map(|v| convert_val_type(*v))
                                        .collect(),
                                    results: func_type
                                        .results()
                                        .iter()
                                        .map(|v| convert_val_type(*v))
                                        .collect(),
                                });
                            } else {
                                // Non-function type (struct/array/cont) — push a placeholder.
                                types.push(FuncSig {
                                    params: vec![],
                                    results: vec![],
                                });
                            }
                        }
                    }
                }

                Payload::ImportSection(reader) => {
                    for imports_group in reader {
                        let imports_group =
                            imports_group.map_err(|e| format!("{label}: import error: {e}"))?;
                        for import_result in imports_group {
                            let (_offset, import) = import_result
                                .map_err(|e| format!("{label}: import item error: {e}"))?;
                            match import.ty {
                                wasmparser::TypeRef::Func(type_idx) => {
                                    func_imports.push(ParsedImport {
                                        module: import.module.to_string(),
                                        name: import.name.to_string(),
                                        type_idx,
                                    });
                                    num_func_imports += 1;
                                }
                                wasmparser::TypeRef::Memory(mem) => {
                                    memory = Some((mem.initial, mem.maximum));
                                }
                                _ => {}
                            }
                        }
                    }
                }

                Payload::FunctionSection(reader) => {
                    for type_idx in reader {
                        let type_idx =
                            type_idx.map_err(|e| format!("{label}: func section error: {e}"))?;
                        functions.push(ParsedFunction { type_idx });
                    }
                }

                Payload::MemorySection(reader) => {
                    for mem in reader {
                        let mem = mem.map_err(|e| format!("{label}: memory error: {e}"))?;
                        memory = Some((mem.initial, mem.maximum));
                    }
                }

                Payload::GlobalSection(reader) => {
                    for global in reader {
                        let global = global.map_err(|e| format!("{label}: global error: {e}"))?;
                        let vt = convert_val_type(global.ty.content_type);
                        let init = parse_const_expr(&global.init_expr);
                        globals.push(ParsedGlobal {
                            val_type: vt,
                            mutable: global.ty.mutable,
                            init,
                        });
                    }
                }

                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export.map_err(|e| format!("{label}: export error: {e}"))?;
                        let kind = match export.kind {
                            wasmparser::ExternalKind::Func => ExportItemKind::Func,
                            wasmparser::ExternalKind::Memory => ExportItemKind::Memory,
                            wasmparser::ExternalKind::Global => ExportItemKind::Global,
                            wasmparser::ExternalKind::Table => ExportItemKind::Table,
                            _ => continue,
                        };
                        if kind == ExportItemKind::Func {
                            export_func_map.insert(export.name.to_string(), export.index);
                        }
                        exports.push(ParsedExport {
                            name: export.name.to_string(),
                            kind,
                            index: export.index,
                        });
                    }
                }

                Payload::TableSection(reader) => {
                    for table in reader {
                        let table = table.map_err(|e| format!("{label}: table error: {e}"))?;
                        eprintln!(
                            "[wasm-linker] {label}: parsed table initial={}, max={:?}",
                            table.ty.initial, table.ty.maximum
                        );
                        tables.push((table.ty.initial as u32, table.ty.maximum.map(|m| m as u32)));
                    }
                }

                Payload::ElementSection(reader) => {
                    for elem in reader {
                        let elem = elem.map_err(|e| format!("{label}: element error: {e}"))?;
                        if let wasmparser::ElementKind::Active {
                            table_index,
                            offset_expr,
                        } = elem.kind
                        {
                            let offset = match parse_const_expr(&offset_expr) {
                                GlobalInit::I32(v) => v as u32,
                                _ => 0,
                            };
                            let mut func_indices = Vec::new();
                            if let wasmparser::ElementItems::Functions(reader) = elem.items {
                                for idx in reader {
                                    func_indices
                                        .push(idx.map_err(|e| format!("{label}: elem func: {e}"))?);
                                }
                            }
                            elements.push((table_index.unwrap_or(0), offset, func_indices));
                        }
                    }
                }

                Payload::DataSection(reader) => {
                    for data in reader {
                        let data = data.map_err(|e| format!("{label}: data error: {e}"))?;
                        if let wasmparser::DataKind::Active {
                            memory_index,
                            offset_expr,
                        } = data.kind
                        {
                            let offset = match parse_const_expr(&offset_expr) {
                                GlobalInit::I32(v) => v as u32,
                                GlobalInit::I64(v) => v as u32,
                                _ => 0,
                            };
                            data_segments.push(ParsedData {
                                memory_idx: memory_index,
                                offset,
                                data: data.data.to_vec(),
                            });
                        }
                    }
                }

                Payload::CodeSectionEntry(body) => {
                    // Store the raw bytes of the function body for later re-encoding.
                    code_bodies.push(body.as_bytes().to_vec());
                }

                _ => {}
            }
        }

        Ok(ParsedModule {
            types,
            func_imports,
            functions,
            code_bodies,
            exports,
            export_func_map,
            memory,
            globals,
            data_segments,
            num_func_imports,
            tables,
            elements,
        })
    }
}

// ---------------------------------------------------------------------------
// Linker context
// ---------------------------------------------------------------------------

struct LinkerCtx {
    /// Merged type section (deduplicated).
    merged_types: Vec<FuncSig>,
    type_map: BTreeMap<FuncSig, u32>,

    // Index remapping for the runtime module.
    /// runtime original func idx -> merged func idx.
    rt_func_remap: BTreeMap<u32, u32>,
    /// runtime original global idx -> merged global idx.
    rt_global_remap: BTreeMap<u32, u32>,
    /// runtime original type idx -> merged type idx.
    rt_type_remap: BTreeMap<u32, u32>,

    // Index remapping for the user module.
    /// user original func idx -> merged func idx.
    user_func_remap: BTreeMap<u32, u32>,
    /// user original global idx -> merged global idx.
    user_global_remap: BTreeMap<u32, u32>,
    /// user original type idx -> merged type idx.
    user_type_remap: BTreeMap<u32, u32>,

    /// Number of WASI imports in the merged module.
    n_wasi_imports: u32,

    /// Stub function type indices (in merged type section) for each stub.
    stub_type_indices: Vec<u32>,

    /// Names of unresolved imports (for diagnostics).
    unresolved_imports: Vec<String>,

    /// Non-rayzor imports preserved as real WASM imports (from host modules).
    /// These are emitted in the import section so the JS host can provide them.
    preserved_imports: Vec<PreservedImport>,

    /// Host function map: function_name → host_module_name.
    /// "rayzor" imports matching this map are preserved instead of stubbed.
    host_functions: BTreeMap<String, String>,
    /// Total merged function count (set after merge).
    total_func_count: u32,
}

/// A user import that's preserved in the linked output (not stubbed).
/// Used for @:jsImport functions that the JS host module provides.
struct PreservedImport {
    module: String,
    name: String,
    type_idx: u32,
    merged_func_idx: u32,
}

impl LinkerCtx {
    fn new() -> Self {
        Self {
            merged_types: Vec::new(),
            type_map: BTreeMap::new(),
            rt_func_remap: BTreeMap::new(),
            rt_global_remap: BTreeMap::new(),
            rt_type_remap: BTreeMap::new(),
            user_func_remap: BTreeMap::new(),
            user_global_remap: BTreeMap::new(),
            user_type_remap: BTreeMap::new(),
            n_wasi_imports: 0,
            stub_type_indices: Vec::new(),
            unresolved_imports: Vec::new(),
            preserved_imports: Vec::new(),
            host_functions: BTreeMap::new(),
            total_func_count: 0,
        }
    }

    /// Intern a function signature, returning its index in the merged type section.
    fn intern_type(&mut self, sig: &FuncSig) -> u32 {
        if let Some(&idx) = self.type_map.get(sig) {
            return idx;
        }
        let idx = self.merged_types.len() as u32;
        self.merged_types.push(sig.clone());
        self.type_map.insert(sig.clone(), idx);
        idx
    }

    /// Build all remapping tables and determine the merged layout.
    fn merge(&mut self, rt: &ParsedModule, user: &ParsedModule) -> Result<(), String> {
        // --- Step 1: Merge types from both modules ---
        for (i, sig) in rt.types.iter().enumerate() {
            let merged_idx = self.intern_type(sig);
            self.rt_type_remap.insert(i as u32, merged_idx);
        }
        for (i, sig) in user.types.iter().enumerate() {
            let merged_idx = self.intern_type(sig);
            self.user_type_remap.insert(i as u32, merged_idx);
        }

        // --- Step 2: Compute function index layout ---
        let mut next_func_idx: u32 = 0;

        // 2a. WASI imports from runtime (all runtime imports are kept).
        for (i, _imp) in rt.func_imports.iter().enumerate() {
            self.rt_func_remap.insert(i as u32, next_func_idx);
            next_func_idx += 1;
        }
        self.n_wasi_imports = rt.func_imports.len() as u32;

        // 2b. Runtime internal functions.
        for i in 0..rt.functions.len() {
            let rt_orig_idx = rt.num_func_imports + i as u32;
            self.rt_func_remap.insert(rt_orig_idx, next_func_idx);
            next_func_idx += 1;
        }

        // 2c. Resolve user imports against runtime exports.
        // Build runtime export name -> merged function index map.
        let mut rt_export_to_merged: BTreeMap<&str, u32> = BTreeMap::new();
        for export in &rt.exports {
            if export.kind == ExportItemKind::Func {
                if let Some(&merged_idx) = self.rt_func_remap.get(&export.index) {
                    rt_export_to_merged.insert(&export.name, merged_idx);
                }
            }
        }

        // Alias mappings: user imports these names, runtime exports prefixed versions.
        // This avoids libc symbol collisions in the runtime crate and maps
        // MIR stdlib wrapper names (e.g., `array_length`) to their runtime-wasm
        // implementations (e.g., `haxe_array_length`).
        let aliases: &[(&str, &str)] = &[
            ("malloc", "rayzor_malloc"),
            ("free", "rayzor_free"),
            ("realloc", "rayzor_realloc"),
            // Array methods — MIR wrappers use bare `array_*` names
            ("array_length", "haxe_array_length"),
            ("array_push", "haxe_array_push_i64"),
            ("array_push_f64", "haxe_array_push_f64"),
            ("array_pop", "haxe_array_pop_i64"),
            ("array_get", "haxe_array_get_i64"),
            ("array_set", "haxe_array_set_i64"),
            ("array_concat", "haxe_array_concat"),
            ("array_copy", "haxe_array_copy"),
            ("array_reverse", "haxe_array_reverse"),
            ("array_slice", "haxe_array_slice"),
            ("array_join", "haxe_array_join"),
            ("array_index_of", "haxe_array_index_of"),
            ("array_last_index_of", "haxe_array_last_index_of"),
            ("array_contains", "haxe_array_contains"),
            ("array_shift", "haxe_array_shift"),
            ("array_unshift", "haxe_array_unshift"),
            ("array_splice", "haxe_array_splice"),
            ("array_resize", "haxe_array_resize"),
            ("array_filter", "haxe_array_filter"),
            ("array_map", "haxe_array_map"),
        ];
        for &(user_name, rt_name) in aliases {
            if let Some(&idx) = rt_export_to_merged.get(rt_name) {
                rt_export_to_merged.insert(user_name, idx);
            }
        }

        // Two-pass user import resolution:
        // Pass A: classify each user import as resolved, stubbed, or preserved.
        enum UserImportKind {
            Resolved(u32),                  // Points to runtime function (merged_idx)
            Stub(u32),                      // Stub type index
            Preserved(String, String, u32), // (module, name, type_idx)
        }
        let mut user_import_kinds: Vec<UserImportKind> = Vec::new();

        for (_i, imp) in user.func_imports.iter().enumerate() {
            if imp.module == "rayzor" {
                if let Some(&merged_idx) = rt_export_to_merged.get(imp.name.as_str()) {
                    let user_type = self
                        .user_type_remap
                        .get(&imp.type_idx)
                        .copied()
                        .unwrap_or(0);
                    let rt_internal_idx = if merged_idx >= self.n_wasi_imports {
                        (merged_idx - self.n_wasi_imports) as usize
                    } else {
                        0
                    };
                    let rt_type = if rt_internal_idx < rt.functions.len() {
                        self.rt_type_remap
                            .get(&rt.functions[rt_internal_idx].type_idx)
                            .copied()
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    if user_type != rt_type {
                        let merged_type_idx = self
                            .user_type_remap
                            .get(&imp.type_idx)
                            .copied()
                            .unwrap_or(0);
                        self.unresolved_imports
                            .push(format!("{}(sig mismatch)", imp.name));
                        user_import_kinds.push(UserImportKind::Stub(merged_type_idx));
                    } else {
                        user_import_kinds.push(UserImportKind::Resolved(merged_idx));
                    }
                } else if let Some(host_module) = self.host_functions.get(&imp.name) {
                    // This "rayzor" import is provided by an external host module
                    let type_idx = self
                        .user_type_remap
                        .get(&imp.type_idx)
                        .copied()
                        .unwrap_or(0);
                    user_import_kinds.push(UserImportKind::Preserved(
                        host_module.clone(),
                        imp.name.clone(),
                        type_idx,
                    ));
                } else {
                    let type_idx = self
                        .user_type_remap
                        .get(&imp.type_idx)
                        .copied()
                        .unwrap_or(0);
                    // Preserve ALL unresolved "rayzor" imports as JS runtime imports.
                    // The JS Proxy provides default implementations (return 0) for any
                    // unknown function. This handles both first-party and third-party
                    // rpkg imports without a hardcoded whitelist.
                    self.unresolved_imports.push(imp.name.clone());
                    user_import_kinds.push(UserImportKind::Preserved(
                        "rayzor".to_string(),
                        imp.name.clone(),
                        type_idx,
                    ));
                }
            } else {
                let type_idx = self
                    .user_type_remap
                    .get(&imp.type_idx)
                    .copied()
                    .unwrap_or(0);
                user_import_kinds.push(UserImportKind::Preserved(
                    imp.module.clone(),
                    imp.name.clone(),
                    type_idx,
                ));
            }
        }

        // Count preserved imports — they go in the import section after WASI imports.
        let n_preserved = user_import_kinds
            .iter()
            .filter(|k| matches!(k, UserImportKind::Preserved(..)))
            .count() as u32;

        // Shift ALL internal function indices to make room for preserved imports.
        // Imports: [WASI: 0..n_wasi] [Preserved: n_wasi..n_wasi+n_preserved]
        // Internals: [n_wasi+n_preserved..]
        if n_preserved > 0 {
            // Shift RT function remap
            for (_k, v) in self.rt_func_remap.iter_mut() {
                if *v >= self.n_wasi_imports {
                    *v += n_preserved;
                }
            }
            // Also shift rt_export_to_merged
            for (_k, v) in rt_export_to_merged.iter_mut() {
                if *v >= self.n_wasi_imports {
                    *v += n_preserved;
                }
            }
            next_func_idx += n_preserved;
        }

        // Pass B: assign final indices.
        let mut user_import_resolved_count = 0u32;
        let mut preserved_slot = self.n_wasi_imports; // preserved imports start right after WASI
        for (i, kind) in user_import_kinds.into_iter().enumerate() {
            let user_orig_idx = i as u32;
            match kind {
                UserImportKind::Resolved(mut merged_idx) => {
                    // Shift if needed (resolved points to RT function which was shifted)
                    if n_preserved > 0 && merged_idx >= self.n_wasi_imports {
                        merged_idx += n_preserved;
                    }
                    self.user_func_remap.insert(user_orig_idx, merged_idx);
                    user_import_resolved_count += 1;
                }
                UserImportKind::Stub(type_idx) => {
                    self.user_func_remap.insert(user_orig_idx, next_func_idx);
                    self.stub_type_indices.push(type_idx);
                    next_func_idx += 1;
                }
                UserImportKind::Preserved(module, name, type_idx) => {
                    self.preserved_imports.push(PreservedImport {
                        module,
                        name,
                        type_idx,
                        merged_func_idx: preserved_slot,
                    });
                    self.user_func_remap.insert(user_orig_idx, preserved_slot);
                    preserved_slot += 1;
                }
            }
        }

        // 2d. User internal functions.
        let n_resolved = user_import_resolved_count;
        eprintln!(
            "[wasm-linker] user imports: {} total ({} resolved, {} preserved, {} stubs), {} internal funcs",
            user.num_func_imports,
            n_resolved,
            self.preserved_imports.len(),
            self.stub_type_indices.len(),
            user.functions.len()
        );
        for i in 0..user.functions.len() {
            let user_orig_idx = user.num_func_imports + i as u32;
            self.user_func_remap.insert(user_orig_idx, next_func_idx);
            next_func_idx += 1;
        }

        // --- Step 3: Global index remapping ---
        // Runtime globals keep their indices (0..N_rt_globals).
        for i in 0..rt.globals.len() {
            self.rt_global_remap.insert(i as u32, i as u32);
        }
        // User globals are shifted after runtime globals.
        let rt_globals_count = rt.globals.len() as u32;
        for i in 0..user.globals.len() {
            self.user_global_remap
                .insert(i as u32, rt_globals_count + i as u32);
        }

        if !self.unresolved_imports.is_empty() {
            eprintln!(
                "[wasm-linker] {} unresolved imports (stubs generated):",
                self.unresolved_imports.len()
            );
            for name in &self.unresolved_imports {
                eprintln!("  - {name}");
            }
        }

        self.total_func_count = next_func_idx;
        Ok(())
    }

    /// Encode the merged module.
    fn encode(&self, rt: &ParsedModule, user: &ParsedModule) -> Result<Vec<u8>, String> {
        let mut module = Module::new();

        // --- Type section ---
        let mut type_section = TypeSection::new();
        for sig in &self.merged_types {
            type_section
                .ty()
                .function(sig.params.iter().copied(), sig.results.iter().copied());
        }
        module.section(&type_section);

        // --- Import section (WASI imports from runtime only) ---
        // Build import section: WASI imports first, then preserved @:jsImport imports.
        let mut import_section = ImportSection::new();

        // WASI imports (indices 0..n_wasi)
        for imp in &rt.func_imports {
            let merged_type_idx = self.rt_type_remap.get(&imp.type_idx).copied().unwrap_or(0);
            import_section.import(
                &imp.module,
                &imp.name,
                EntityType::Function(merged_type_idx),
            );
        }

        // Preserved @:jsImport imports
        for imp in &self.preserved_imports {
            import_section.import(&imp.module, &imp.name, EntityType::Function(imp.type_idx));
        }
        let n_preserved = self.preserved_imports.len();
        module.section(&import_section);

        // --- Function section ---
        // Order: runtime internals, stubs, user internals.
        let mut func_section = FunctionSection::new();
        for func in &rt.functions {
            let merged_type_idx = self.rt_type_remap.get(&func.type_idx).copied().unwrap_or(0);
            func_section.function(merged_type_idx);
        }
        for &stub_type_idx in &self.stub_type_indices {
            func_section.function(stub_type_idx);
        }
        for func in &user.functions {
            let merged_type_idx = self
                .user_type_remap
                .get(&func.type_idx)
                .copied()
                .unwrap_or(0);
            func_section.function(merged_type_idx);
        }
        module.section(&func_section);

        // --- Table section (from runtime, needed for call_indirect) ---
        eprintln!(
            "[wasm-linker] encode: rt.tables.len()={}, rt.elements.len()={}",
            rt.tables.len(),
            rt.elements.len()
        );
        // Grow table to accommodate user closure functions.
        // User closures store raw func indices; they need table entries to be callable
        // via call_indirect and from JS.
        let rt_table_size = rt.tables.first().map(|(init, _)| *init).unwrap_or(0);
        let table_size = rt_table_size.max(self.total_func_count);
        eprintln!("[wasm-linker] table: rt={}, total_funcs={}, merged={}", rt_table_size, self.total_func_count, table_size);
        if !rt.tables.is_empty() || table_size > 0 {
            let mut table_section = wasm_encoder::TableSection::new();
            table_section.table(wasm_encoder::TableType {
                element_type: wasm_encoder::RefType::FUNCREF,
                minimum: table_size as u64,
                maximum: Some(table_size as u64),
                table64: false,
                shared: false,
            });
            module.section(&table_section);
        }

        // --- Memory section ---
        let rt_mem = rt.memory.unwrap_or((256, None));
        let user_mem = user.memory.unwrap_or((256, None));
        // Ensure minimum 512 pages (32 MB) for heap room above data/stack sections.
        // The runtime's __heap_base can be ~16 MB, so 256 pages (16 MB) leaves no heap.
        let merged_initial = rt_mem.0.max(user_mem.0).max(512);
        let merged_maximum = match (rt_mem.1, user_mem.1) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            // Allow growth up to 1024 pages (64 MB) by default
            (None, None) => Some(1024),
        };
        let mut mem_section = MemorySection::new();
        mem_section.memory(EncMemoryType {
            minimum: merged_initial,
            maximum: merged_maximum,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mem_section);

        // --- Global section ---
        let mut global_section = GlobalSection::new();
        for g in &rt.globals {
            let enc_init = encode_global_init(&g.init, &self.rt_global_remap);
            global_section.global(
                EncGlobalType {
                    val_type: g.val_type,
                    mutable: g.mutable,
                    shared: false,
                },
                &enc_init,
            );
        }
        for g in &user.globals {
            let enc_init = encode_global_init(&g.init, &self.user_global_remap);
            global_section.global(
                EncGlobalType {
                    val_type: g.val_type,
                    mutable: g.mutable,
                    shared: false,
                },
                &enc_init,
            );
        }
        module.section(&global_section);

        // --- Export section ---
        let mut export_section = ExportSection::new();
        export_section.export("memory", ExportKind::Memory, 0);
        export_section.export("__indirect_function_table", ExportKind::Table, 0);
        // Preserve ALL user exports (not just _start)
        for export in &user.exports {
            if export.kind == ExportItemKind::Func {
                if let Some(&merged_idx) = self.user_func_remap.get(&export.index) {
                    export_section.export(&export.name, ExportKind::Func, merged_idx);
                }
            }
        }
        module.section(&export_section);

        // --- Element section (from runtime, populates function table) ---
        {
            let mut elem_section = wasm_encoder::ElementSection::new();
            // Runtime element entries (remapped)
            for (table_idx, offset, func_indices) in &rt.elements {
                let remapped: Vec<u32> = func_indices
                    .iter()
                    .map(|&idx| self.rt_func_remap.get(&idx).copied().unwrap_or(idx))
                    .collect();
                elem_section.active(
                    Some(*table_idx),
                    &ConstExpr::i32_const(*offset as i32),
                    wasm_encoder::Elements::Functions(std::borrow::Cow::Owned(remapped)),
                );
            }
            // Add ALL user internal functions to table (for closures / call_indirect).
            // User functions start after imports in the merged index space.
            // Map each user-module function to its merged index.
            let user_func_start = self.n_wasi_imports + n_preserved as u32 + rt.functions.len() as u32 + self.stub_type_indices.len() as u32;
            let user_func_count = user.functions.len() as u32;
            if user_func_count > 0 {
                let user_funcs: Vec<u32> = (0..user_func_count)
                    .map(|i| user_func_start + i)
                    .collect();
                elem_section.active(
                    Some(0),
                    &ConstExpr::i32_const(user_func_start as i32),
                    wasm_encoder::Elements::Functions(std::borrow::Cow::Owned(user_funcs)),
                );
            }
            module.section(&elem_section);
        }

        // --- Code section ---
        let mut code_section = CodeSection::new();

        // Runtime internal function bodies.
        for body_bytes in &rt.code_bodies {
            let func = reencode_function_body(
                body_bytes,
                &self.rt_func_remap,
                &self.rt_global_remap,
                &self.rt_type_remap,
                "runtime",
            )?;
            code_section.function(&func);
        }

        // Stub function bodies (return default value or unreachable).
        for &stub_type_idx in &self.stub_type_indices {
            let sig = &self.merged_types[stub_type_idx as usize];
            let mut func = Function::new(std::iter::empty::<(u32, ValType)>());
            if sig.results.is_empty() {
                func.instruction(&Instruction::Return);
            } else {
                for result_ty in &sig.results {
                    match result_ty {
                        ValType::I32 => func.instruction(&Instruction::I32Const(0)),
                        ValType::I64 => func.instruction(&Instruction::I64Const(0)),
                        ValType::F32 => func.instruction(&Instruction::F32Const(
                            wasm_encoder::Ieee32::from(0.0f32),
                        )),
                        ValType::F64 => func.instruction(&Instruction::F64Const(
                            wasm_encoder::Ieee64::from(0.0f64),
                        )),
                        _ => func.instruction(&Instruction::I32Const(0)),
                    };
                }
                func.instruction(&Instruction::Return);
            }
            func.instruction(&Instruction::End);
            code_section.function(&func);
        }

        // User internal function bodies.
        for body_bytes in &user.code_bodies {
            let func = reencode_function_body(
                body_bytes,
                &self.user_func_remap,
                &self.user_global_remap,
                &self.user_type_remap,
                "user",
            )?;
            code_section.function(&func);
        }

        module.section(&code_section);

        // --- Data section ---
        let mut data_section = DataSection::new();
        for seg in &rt.data_segments {
            data_section.active(
                seg.memory_idx,
                &ConstExpr::i32_const(seg.offset as i32),
                seg.data.iter().copied(),
            );
        }
        for seg in &user.data_segments {
            data_section.active(
                seg.memory_idx,
                &ConstExpr::i32_const(seg.offset as i32),
                seg.data.iter().copied(),
            );
        }
        module.section(&data_section);

        Ok(module.finish())
    }
}

// ---------------------------------------------------------------------------
// Function body re-encoding
// ---------------------------------------------------------------------------

/// Re-encode a function body, rewriting all index references.
///
/// Parses the raw function body bytes with wasmparser, iterates through
/// locals and operators, and re-emits each instruction via wasm-encoder
/// with remapped indices.
fn reencode_function_body(
    body_bytes: &[u8],
    func_remap: &BTreeMap<u32, u32>,
    global_remap: &BTreeMap<u32, u32>,
    type_remap: &BTreeMap<u32, u32>,
    label: &str,
) -> Result<Function, String> {
    use wasmparser::{BinaryReader, FunctionBody};

    let reader = BinaryReader::new(body_bytes, 0);
    let body = FunctionBody::new(reader);

    // Parse locals.
    let locals_reader = body
        .get_locals_reader()
        .map_err(|e| format!("{label}: locals error: {e}"))?;
    let mut local_groups: Vec<(u32, ValType)> = Vec::new();
    for local in locals_reader {
        let (count, vt) = local.map_err(|e| format!("{label}: local read error: {e}"))?;
        local_groups.push((count, convert_val_type(vt)));
    }

    let mut func = Function::new(local_groups);

    // Parse and re-emit operators.
    let ops_reader = body
        .get_operators_reader()
        .map_err(|e| format!("{label}: operators error: {e}"))?;

    for op in ops_reader {
        let op = op.map_err(|e| format!("{label}: operator error: {e}"))?;
        let instruction = translate_operator(&op, func_remap, global_remap, type_remap);
        func.instruction(&instruction);
    }

    Ok(func)
}

/// Translate a single wasmparser `Operator` to a wasm-encoder `Instruction`,
/// remapping function, global, and type indices as needed.
fn translate_operator<'a>(
    op: &wasmparser::Operator<'a>,
    func_remap: &BTreeMap<u32, u32>,
    global_remap: &BTreeMap<u32, u32>,
    type_remap: &BTreeMap<u32, u32>,
) -> Instruction<'static> {
    use wasmparser::Operator as Op;

    match op {
        // ----- Control flow -----
        Op::Unreachable => Instruction::Unreachable,
        Op::Nop => Instruction::Nop,
        Op::Block { blockty } => Instruction::Block(convert_block_type(blockty, type_remap)),
        Op::Loop { blockty } => Instruction::Loop(convert_block_type(blockty, type_remap)),
        Op::If { blockty } => Instruction::If(convert_block_type(blockty, type_remap)),
        Op::Else => Instruction::Else,
        Op::End => Instruction::End,
        Op::Br { relative_depth } => Instruction::Br(*relative_depth),
        Op::BrIf { relative_depth } => Instruction::BrIf(*relative_depth),
        Op::BrTable { targets } => {
            let mut target_vec: Vec<u32> = Vec::new();
            for target in targets.targets() {
                target_vec.push(target.unwrap_or(0));
            }
            let default = targets.default();
            Instruction::BrTable(Cow::Owned(target_vec), default)
        }
        Op::Return => Instruction::Return,

        // ----- Calls (index rewriting) -----
        Op::Call { function_index } => {
            let merged = func_remap
                .get(function_index)
                .copied()
                .unwrap_or(*function_index);
            Instruction::Call(merged)
        }
        Op::CallIndirect {
            type_index,
            table_index,
        } => {
            let merged_type = type_remap.get(type_index).copied().unwrap_or(*type_index);
            Instruction::CallIndirect {
                type_index: merged_type,
                table_index: *table_index,
            }
        }
        Op::ReturnCall { function_index } => {
            let merged = func_remap
                .get(function_index)
                .copied()
                .unwrap_or(*function_index);
            Instruction::ReturnCall(merged)
        }
        Op::ReturnCallIndirect {
            type_index,
            table_index,
        } => {
            let merged_type = type_remap.get(type_index).copied().unwrap_or(*type_index);
            Instruction::ReturnCallIndirect {
                type_index: merged_type,
                table_index: *table_index,
            }
        }
        Op::RefFunc { function_index } => {
            let merged = func_remap
                .get(function_index)
                .copied()
                .unwrap_or(*function_index);
            Instruction::RefFunc(merged)
        }

        // ----- Locals -----
        Op::LocalGet { local_index } => Instruction::LocalGet(*local_index),
        Op::LocalSet { local_index } => Instruction::LocalSet(*local_index),
        Op::LocalTee { local_index } => Instruction::LocalTee(*local_index),

        // ----- Globals (index rewriting) -----
        Op::GlobalGet { global_index } => {
            let merged = global_remap
                .get(global_index)
                .copied()
                .unwrap_or(*global_index);
            Instruction::GlobalGet(merged)
        }
        Op::GlobalSet { global_index } => {
            let merged = global_remap
                .get(global_index)
                .copied()
                .unwrap_or(*global_index);
            Instruction::GlobalSet(merged)
        }

        // ----- Memory loads -----
        Op::I32Load { memarg } => Instruction::I32Load(convert_memarg(memarg)),
        Op::I64Load { memarg } => Instruction::I64Load(convert_memarg(memarg)),
        Op::F32Load { memarg } => Instruction::F32Load(convert_memarg(memarg)),
        Op::F64Load { memarg } => Instruction::F64Load(convert_memarg(memarg)),
        Op::I32Load8S { memarg } => Instruction::I32Load8S(convert_memarg(memarg)),
        Op::I32Load8U { memarg } => Instruction::I32Load8U(convert_memarg(memarg)),
        Op::I32Load16S { memarg } => Instruction::I32Load16S(convert_memarg(memarg)),
        Op::I32Load16U { memarg } => Instruction::I32Load16U(convert_memarg(memarg)),
        Op::I64Load8S { memarg } => Instruction::I64Load8S(convert_memarg(memarg)),
        Op::I64Load8U { memarg } => Instruction::I64Load8U(convert_memarg(memarg)),
        Op::I64Load16S { memarg } => Instruction::I64Load16S(convert_memarg(memarg)),
        Op::I64Load16U { memarg } => Instruction::I64Load16U(convert_memarg(memarg)),
        Op::I64Load32S { memarg } => Instruction::I64Load32S(convert_memarg(memarg)),
        Op::I64Load32U { memarg } => Instruction::I64Load32U(convert_memarg(memarg)),

        // ----- Memory stores -----
        Op::I32Store { memarg } => Instruction::I32Store(convert_memarg(memarg)),
        Op::I64Store { memarg } => Instruction::I64Store(convert_memarg(memarg)),
        Op::F32Store { memarg } => Instruction::F32Store(convert_memarg(memarg)),
        Op::F64Store { memarg } => Instruction::F64Store(convert_memarg(memarg)),
        Op::I32Store8 { memarg } => Instruction::I32Store8(convert_memarg(memarg)),
        Op::I32Store16 { memarg } => Instruction::I32Store16(convert_memarg(memarg)),
        Op::I64Store8 { memarg } => Instruction::I64Store8(convert_memarg(memarg)),
        Op::I64Store16 { memarg } => Instruction::I64Store16(convert_memarg(memarg)),
        Op::I64Store32 { memarg } => Instruction::I64Store32(convert_memarg(memarg)),

        // ----- Memory operations -----
        Op::MemorySize { mem } => Instruction::MemorySize(*mem),
        Op::MemoryGrow { mem } => Instruction::MemoryGrow(*mem),
        Op::MemoryInit { data_index, mem } => Instruction::MemoryInit {
            data_index: *data_index,
            mem: *mem,
        },
        Op::DataDrop { data_index } => Instruction::DataDrop(*data_index),
        Op::MemoryCopy { dst_mem, src_mem } => Instruction::MemoryCopy {
            src_mem: *src_mem,
            dst_mem: *dst_mem,
        },
        Op::MemoryFill { mem } => Instruction::MemoryFill(*mem),

        // ----- Constants -----
        Op::I32Const { value } => Instruction::I32Const(*value),
        Op::I64Const { value } => Instruction::I64Const(*value),
        Op::F32Const { value } => Instruction::F32Const(wasm_encoder::Ieee32::new(value.bits())),
        Op::F64Const { value } => Instruction::F64Const(wasm_encoder::Ieee64::new(value.bits())),

        // ----- Comparison i32 -----
        Op::I32Eqz => Instruction::I32Eqz,
        Op::I32Eq => Instruction::I32Eq,
        Op::I32Ne => Instruction::I32Ne,
        Op::I32LtS => Instruction::I32LtS,
        Op::I32LtU => Instruction::I32LtU,
        Op::I32GtS => Instruction::I32GtS,
        Op::I32GtU => Instruction::I32GtU,
        Op::I32LeS => Instruction::I32LeS,
        Op::I32LeU => Instruction::I32LeU,
        Op::I32GeS => Instruction::I32GeS,
        Op::I32GeU => Instruction::I32GeU,

        // ----- Comparison i64 -----
        Op::I64Eqz => Instruction::I64Eqz,
        Op::I64Eq => Instruction::I64Eq,
        Op::I64Ne => Instruction::I64Ne,
        Op::I64LtS => Instruction::I64LtS,
        Op::I64LtU => Instruction::I64LtU,
        Op::I64GtS => Instruction::I64GtS,
        Op::I64GtU => Instruction::I64GtU,
        Op::I64LeS => Instruction::I64LeS,
        Op::I64LeU => Instruction::I64LeU,
        Op::I64GeS => Instruction::I64GeS,
        Op::I64GeU => Instruction::I64GeU,

        // ----- Comparison f32 -----
        Op::F32Eq => Instruction::F32Eq,
        Op::F32Ne => Instruction::F32Ne,
        Op::F32Lt => Instruction::F32Lt,
        Op::F32Gt => Instruction::F32Gt,
        Op::F32Le => Instruction::F32Le,
        Op::F32Ge => Instruction::F32Ge,

        // ----- Comparison f64 -----
        Op::F64Eq => Instruction::F64Eq,
        Op::F64Ne => Instruction::F64Ne,
        Op::F64Lt => Instruction::F64Lt,
        Op::F64Gt => Instruction::F64Gt,
        Op::F64Le => Instruction::F64Le,
        Op::F64Ge => Instruction::F64Ge,

        // ----- I32 arithmetic -----
        Op::I32Clz => Instruction::I32Clz,
        Op::I32Ctz => Instruction::I32Ctz,
        Op::I32Popcnt => Instruction::I32Popcnt,
        Op::I32Add => Instruction::I32Add,
        Op::I32Sub => Instruction::I32Sub,
        Op::I32Mul => Instruction::I32Mul,
        Op::I32DivS => Instruction::I32DivS,
        Op::I32DivU => Instruction::I32DivU,
        Op::I32RemS => Instruction::I32RemS,
        Op::I32RemU => Instruction::I32RemU,
        Op::I32And => Instruction::I32And,
        Op::I32Or => Instruction::I32Or,
        Op::I32Xor => Instruction::I32Xor,
        Op::I32Shl => Instruction::I32Shl,
        Op::I32ShrS => Instruction::I32ShrS,
        Op::I32ShrU => Instruction::I32ShrU,
        Op::I32Rotl => Instruction::I32Rotl,
        Op::I32Rotr => Instruction::I32Rotr,

        // ----- I64 arithmetic -----
        Op::I64Clz => Instruction::I64Clz,
        Op::I64Ctz => Instruction::I64Ctz,
        Op::I64Popcnt => Instruction::I64Popcnt,
        Op::I64Add => Instruction::I64Add,
        Op::I64Sub => Instruction::I64Sub,
        Op::I64Mul => Instruction::I64Mul,
        Op::I64DivS => Instruction::I64DivS,
        Op::I64DivU => Instruction::I64DivU,
        Op::I64RemS => Instruction::I64RemS,
        Op::I64RemU => Instruction::I64RemU,
        Op::I64And => Instruction::I64And,
        Op::I64Or => Instruction::I64Or,
        Op::I64Xor => Instruction::I64Xor,
        Op::I64Shl => Instruction::I64Shl,
        Op::I64ShrS => Instruction::I64ShrS,
        Op::I64ShrU => Instruction::I64ShrU,
        Op::I64Rotl => Instruction::I64Rotl,
        Op::I64Rotr => Instruction::I64Rotr,

        // ----- F32 arithmetic -----
        Op::F32Abs => Instruction::F32Abs,
        Op::F32Neg => Instruction::F32Neg,
        Op::F32Ceil => Instruction::F32Ceil,
        Op::F32Floor => Instruction::F32Floor,
        Op::F32Trunc => Instruction::F32Trunc,
        Op::F32Nearest => Instruction::F32Nearest,
        Op::F32Sqrt => Instruction::F32Sqrt,
        Op::F32Add => Instruction::F32Add,
        Op::F32Sub => Instruction::F32Sub,
        Op::F32Mul => Instruction::F32Mul,
        Op::F32Div => Instruction::F32Div,
        Op::F32Min => Instruction::F32Min,
        Op::F32Max => Instruction::F32Max,
        Op::F32Copysign => Instruction::F32Copysign,

        // ----- F64 arithmetic -----
        Op::F64Abs => Instruction::F64Abs,
        Op::F64Neg => Instruction::F64Neg,
        Op::F64Ceil => Instruction::F64Ceil,
        Op::F64Floor => Instruction::F64Floor,
        Op::F64Trunc => Instruction::F64Trunc,
        Op::F64Nearest => Instruction::F64Nearest,
        Op::F64Sqrt => Instruction::F64Sqrt,
        Op::F64Add => Instruction::F64Add,
        Op::F64Sub => Instruction::F64Sub,
        Op::F64Mul => Instruction::F64Mul,
        Op::F64Div => Instruction::F64Div,
        Op::F64Min => Instruction::F64Min,
        Op::F64Max => Instruction::F64Max,
        Op::F64Copysign => Instruction::F64Copysign,

        // ----- Conversions -----
        Op::I32WrapI64 => Instruction::I32WrapI64,
        Op::I32TruncF32S => Instruction::I32TruncF32S,
        Op::I32TruncF32U => Instruction::I32TruncF32U,
        Op::I32TruncF64S => Instruction::I32TruncF64S,
        Op::I32TruncF64U => Instruction::I32TruncF64U,
        Op::I64ExtendI32S => Instruction::I64ExtendI32S,
        Op::I64ExtendI32U => Instruction::I64ExtendI32U,
        Op::I64TruncF32S => Instruction::I64TruncF32S,
        Op::I64TruncF32U => Instruction::I64TruncF32U,
        Op::I64TruncF64S => Instruction::I64TruncF64S,
        Op::I64TruncF64U => Instruction::I64TruncF64U,
        Op::F32ConvertI32S => Instruction::F32ConvertI32S,
        Op::F32ConvertI32U => Instruction::F32ConvertI32U,
        Op::F32ConvertI64S => Instruction::F32ConvertI64S,
        Op::F32ConvertI64U => Instruction::F32ConvertI64U,
        Op::F32DemoteF64 => Instruction::F32DemoteF64,
        Op::F64ConvertI32S => Instruction::F64ConvertI32S,
        Op::F64ConvertI32U => Instruction::F64ConvertI32U,
        Op::F64ConvertI64S => Instruction::F64ConvertI64S,
        Op::F64ConvertI64U => Instruction::F64ConvertI64U,
        Op::F64PromoteF32 => Instruction::F64PromoteF32,
        Op::I32ReinterpretF32 => Instruction::I32ReinterpretF32,
        Op::I64ReinterpretF64 => Instruction::I64ReinterpretF64,
        Op::F32ReinterpretI32 => Instruction::F32ReinterpretI32,
        Op::F64ReinterpretI64 => Instruction::F64ReinterpretI64,

        // ----- Sign extension -----
        Op::I32Extend8S => Instruction::I32Extend8S,
        Op::I32Extend16S => Instruction::I32Extend16S,
        Op::I64Extend8S => Instruction::I64Extend8S,
        Op::I64Extend16S => Instruction::I64Extend16S,
        Op::I64Extend32S => Instruction::I64Extend32S,

        // ----- Saturating truncation -----
        Op::I32TruncSatF32S => Instruction::I32TruncSatF32S,
        Op::I32TruncSatF32U => Instruction::I32TruncSatF32U,
        Op::I32TruncSatF64S => Instruction::I32TruncSatF64S,
        Op::I32TruncSatF64U => Instruction::I32TruncSatF64U,
        Op::I64TruncSatF32S => Instruction::I64TruncSatF32S,
        Op::I64TruncSatF32U => Instruction::I64TruncSatF32U,
        Op::I64TruncSatF64S => Instruction::I64TruncSatF64S,
        Op::I64TruncSatF64U => Instruction::I64TruncSatF64U,

        // ----- Stack manipulation -----
        Op::Drop => Instruction::Drop,
        Op::Select => Instruction::Select,
        Op::TypedSelect { ty } => Instruction::TypedSelect(convert_val_type(*ty)),

        // ----- Reference types -----
        Op::RefNull { hty } => {
            let enc_ht = convert_heap_type(hty);
            Instruction::RefNull(enc_ht)
        }
        Op::RefIsNull => Instruction::RefIsNull,

        // ----- Table operations -----
        Op::TableGet { table } => Instruction::TableGet(*table),
        Op::TableSet { table } => Instruction::TableSet(*table),
        Op::TableGrow { table } => Instruction::TableGrow(*table),
        Op::TableSize { table } => Instruction::TableSize(*table),
        Op::TableFill { table } => Instruction::TableFill(*table),
        Op::TableCopy {
            dst_table,
            src_table,
        } => Instruction::TableCopy {
            src_table: *src_table,
            dst_table: *dst_table,
        },
        Op::TableInit { elem_index, table } => Instruction::TableInit {
            elem_index: *elem_index,
            table: *table,
        },
        Op::ElemDrop { elem_index } => Instruction::ElemDrop(*elem_index),

        // ----- Atomic fence -----
        Op::AtomicFence => Instruction::AtomicFence,

        // ----- Atomic memory operations -----
        Op::MemoryAtomicNotify { memarg } => {
            Instruction::MemoryAtomicNotify(convert_memarg(memarg))
        }
        Op::MemoryAtomicWait32 { memarg } => {
            Instruction::MemoryAtomicWait32(convert_memarg(memarg))
        }
        Op::MemoryAtomicWait64 { memarg } => {
            Instruction::MemoryAtomicWait64(convert_memarg(memarg))
        }

        // ----- Atomic loads -----
        Op::I32AtomicLoad { memarg } => Instruction::I32AtomicLoad(convert_memarg(memarg)),
        Op::I64AtomicLoad { memarg } => Instruction::I64AtomicLoad(convert_memarg(memarg)),
        Op::I32AtomicLoad8U { memarg } => Instruction::I32AtomicLoad8U(convert_memarg(memarg)),
        Op::I32AtomicLoad16U { memarg } => Instruction::I32AtomicLoad16U(convert_memarg(memarg)),
        Op::I64AtomicLoad8U { memarg } => Instruction::I64AtomicLoad8U(convert_memarg(memarg)),
        Op::I64AtomicLoad16U { memarg } => Instruction::I64AtomicLoad16U(convert_memarg(memarg)),
        Op::I64AtomicLoad32U { memarg } => Instruction::I64AtomicLoad32U(convert_memarg(memarg)),

        // ----- Atomic stores -----
        Op::I32AtomicStore { memarg } => Instruction::I32AtomicStore(convert_memarg(memarg)),
        Op::I64AtomicStore { memarg } => Instruction::I64AtomicStore(convert_memarg(memarg)),
        Op::I32AtomicStore8 { memarg } => Instruction::I32AtomicStore8(convert_memarg(memarg)),
        Op::I32AtomicStore16 { memarg } => Instruction::I32AtomicStore16(convert_memarg(memarg)),
        Op::I64AtomicStore8 { memarg } => Instruction::I64AtomicStore8(convert_memarg(memarg)),
        Op::I64AtomicStore16 { memarg } => Instruction::I64AtomicStore16(convert_memarg(memarg)),
        Op::I64AtomicStore32 { memarg } => Instruction::I64AtomicStore32(convert_memarg(memarg)),

        // ----- Atomic RMW add -----
        Op::I32AtomicRmwAdd { memarg } => Instruction::I32AtomicRmwAdd(convert_memarg(memarg)),
        Op::I64AtomicRmwAdd { memarg } => Instruction::I64AtomicRmwAdd(convert_memarg(memarg)),
        Op::I32AtomicRmw8AddU { memarg } => Instruction::I32AtomicRmw8AddU(convert_memarg(memarg)),
        Op::I32AtomicRmw16AddU { memarg } => {
            Instruction::I32AtomicRmw16AddU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw8AddU { memarg } => Instruction::I64AtomicRmw8AddU(convert_memarg(memarg)),
        Op::I64AtomicRmw16AddU { memarg } => {
            Instruction::I64AtomicRmw16AddU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw32AddU { memarg } => {
            Instruction::I64AtomicRmw32AddU(convert_memarg(memarg))
        }

        // ----- Atomic RMW sub -----
        Op::I32AtomicRmwSub { memarg } => Instruction::I32AtomicRmwSub(convert_memarg(memarg)),
        Op::I64AtomicRmwSub { memarg } => Instruction::I64AtomicRmwSub(convert_memarg(memarg)),
        Op::I32AtomicRmw8SubU { memarg } => Instruction::I32AtomicRmw8SubU(convert_memarg(memarg)),
        Op::I32AtomicRmw16SubU { memarg } => {
            Instruction::I32AtomicRmw16SubU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw8SubU { memarg } => Instruction::I64AtomicRmw8SubU(convert_memarg(memarg)),
        Op::I64AtomicRmw16SubU { memarg } => {
            Instruction::I64AtomicRmw16SubU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw32SubU { memarg } => {
            Instruction::I64AtomicRmw32SubU(convert_memarg(memarg))
        }

        // ----- Atomic RMW and -----
        Op::I32AtomicRmwAnd { memarg } => Instruction::I32AtomicRmwAnd(convert_memarg(memarg)),
        Op::I64AtomicRmwAnd { memarg } => Instruction::I64AtomicRmwAnd(convert_memarg(memarg)),
        Op::I32AtomicRmw8AndU { memarg } => Instruction::I32AtomicRmw8AndU(convert_memarg(memarg)),
        Op::I32AtomicRmw16AndU { memarg } => {
            Instruction::I32AtomicRmw16AndU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw8AndU { memarg } => Instruction::I64AtomicRmw8AndU(convert_memarg(memarg)),
        Op::I64AtomicRmw16AndU { memarg } => {
            Instruction::I64AtomicRmw16AndU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw32AndU { memarg } => {
            Instruction::I64AtomicRmw32AndU(convert_memarg(memarg))
        }

        // ----- Atomic RMW or -----
        Op::I32AtomicRmwOr { memarg } => Instruction::I32AtomicRmwOr(convert_memarg(memarg)),
        Op::I64AtomicRmwOr { memarg } => Instruction::I64AtomicRmwOr(convert_memarg(memarg)),
        Op::I32AtomicRmw8OrU { memarg } => Instruction::I32AtomicRmw8OrU(convert_memarg(memarg)),
        Op::I32AtomicRmw16OrU { memarg } => Instruction::I32AtomicRmw16OrU(convert_memarg(memarg)),
        Op::I64AtomicRmw8OrU { memarg } => Instruction::I64AtomicRmw8OrU(convert_memarg(memarg)),
        Op::I64AtomicRmw16OrU { memarg } => Instruction::I64AtomicRmw16OrU(convert_memarg(memarg)),
        Op::I64AtomicRmw32OrU { memarg } => Instruction::I64AtomicRmw32OrU(convert_memarg(memarg)),

        // ----- Atomic RMW xor -----
        Op::I32AtomicRmwXor { memarg } => Instruction::I32AtomicRmwXor(convert_memarg(memarg)),
        Op::I64AtomicRmwXor { memarg } => Instruction::I64AtomicRmwXor(convert_memarg(memarg)),
        Op::I32AtomicRmw8XorU { memarg } => Instruction::I32AtomicRmw8XorU(convert_memarg(memarg)),
        Op::I32AtomicRmw16XorU { memarg } => {
            Instruction::I32AtomicRmw16XorU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw8XorU { memarg } => Instruction::I64AtomicRmw8XorU(convert_memarg(memarg)),
        Op::I64AtomicRmw16XorU { memarg } => {
            Instruction::I64AtomicRmw16XorU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw32XorU { memarg } => {
            Instruction::I64AtomicRmw32XorU(convert_memarg(memarg))
        }

        // ----- Atomic RMW xchg -----
        Op::I32AtomicRmwXchg { memarg } => Instruction::I32AtomicRmwXchg(convert_memarg(memarg)),
        Op::I64AtomicRmwXchg { memarg } => Instruction::I64AtomicRmwXchg(convert_memarg(memarg)),
        Op::I32AtomicRmw8XchgU { memarg } => {
            Instruction::I32AtomicRmw8XchgU(convert_memarg(memarg))
        }
        Op::I32AtomicRmw16XchgU { memarg } => {
            Instruction::I32AtomicRmw16XchgU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw8XchgU { memarg } => {
            Instruction::I64AtomicRmw8XchgU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw16XchgU { memarg } => {
            Instruction::I64AtomicRmw16XchgU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw32XchgU { memarg } => {
            Instruction::I64AtomicRmw32XchgU(convert_memarg(memarg))
        }

        // ----- Atomic RMW cmpxchg -----
        Op::I32AtomicRmwCmpxchg { memarg } => {
            Instruction::I32AtomicRmwCmpxchg(convert_memarg(memarg))
        }
        Op::I64AtomicRmwCmpxchg { memarg } => {
            Instruction::I64AtomicRmwCmpxchg(convert_memarg(memarg))
        }
        Op::I32AtomicRmw8CmpxchgU { memarg } => {
            Instruction::I32AtomicRmw8CmpxchgU(convert_memarg(memarg))
        }
        Op::I32AtomicRmw16CmpxchgU { memarg } => {
            Instruction::I32AtomicRmw16CmpxchgU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw8CmpxchgU { memarg } => {
            Instruction::I64AtomicRmw8CmpxchgU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw16CmpxchgU { memarg } => {
            Instruction::I64AtomicRmw16CmpxchgU(convert_memarg(memarg))
        }
        Op::I64AtomicRmw32CmpxchgU { memarg } => {
            Instruction::I64AtomicRmw32CmpxchgU(convert_memarg(memarg))
        }

        // ----- SIMD V128 -----
        Op::V128Const { value } => Instruction::V128Const(value.i128() as i128),
        Op::V128Load { memarg } => Instruction::V128Load(convert_memarg(memarg)),
        Op::V128Store { memarg } => Instruction::V128Store(convert_memarg(memarg)),
        Op::V128Not => Instruction::V128Not,
        Op::V128And => Instruction::V128And,
        Op::V128AndNot => Instruction::V128AndNot,
        Op::V128Or => Instruction::V128Or,
        Op::V128Xor => Instruction::V128Xor,
        Op::V128Bitselect => Instruction::V128Bitselect,
        Op::V128AnyTrue => Instruction::V128AnyTrue,

        // ----- SIMD splats -----
        Op::I8x16Splat => Instruction::I8x16Splat,
        Op::I16x8Splat => Instruction::I16x8Splat,
        Op::I32x4Splat => Instruction::I32x4Splat,
        Op::I64x2Splat => Instruction::I64x2Splat,
        Op::F32x4Splat => Instruction::F32x4Splat,
        Op::F64x2Splat => Instruction::F64x2Splat,

        // ----- SIMD extract/replace lanes -----
        Op::I8x16ExtractLaneS { lane } => Instruction::I8x16ExtractLaneS(*lane),
        Op::I8x16ExtractLaneU { lane } => Instruction::I8x16ExtractLaneU(*lane),
        Op::I8x16ReplaceLane { lane } => Instruction::I8x16ReplaceLane(*lane),
        Op::I16x8ExtractLaneS { lane } => Instruction::I16x8ExtractLaneS(*lane),
        Op::I16x8ExtractLaneU { lane } => Instruction::I16x8ExtractLaneU(*lane),
        Op::I16x8ReplaceLane { lane } => Instruction::I16x8ReplaceLane(*lane),
        Op::I32x4ExtractLane { lane } => Instruction::I32x4ExtractLane(*lane),
        Op::I32x4ReplaceLane { lane } => Instruction::I32x4ReplaceLane(*lane),
        Op::I64x2ExtractLane { lane } => Instruction::I64x2ExtractLane(*lane),
        Op::I64x2ReplaceLane { lane } => Instruction::I64x2ReplaceLane(*lane),
        Op::F32x4ExtractLane { lane } => Instruction::F32x4ExtractLane(*lane),
        Op::F32x4ReplaceLane { lane } => Instruction::F32x4ReplaceLane(*lane),
        Op::F64x2ExtractLane { lane } => Instruction::F64x2ExtractLane(*lane),
        Op::F64x2ReplaceLane { lane } => Instruction::F64x2ReplaceLane(*lane),

        // ----- SIMD i8x16 arithmetic -----
        Op::I8x16Eq => Instruction::I8x16Eq,
        Op::I8x16Ne => Instruction::I8x16Ne,
        Op::I8x16LtS => Instruction::I8x16LtS,
        Op::I8x16LtU => Instruction::I8x16LtU,
        Op::I8x16GtS => Instruction::I8x16GtS,
        Op::I8x16GtU => Instruction::I8x16GtU,
        Op::I8x16LeS => Instruction::I8x16LeS,
        Op::I8x16LeU => Instruction::I8x16LeU,
        Op::I8x16GeS => Instruction::I8x16GeS,
        Op::I8x16GeU => Instruction::I8x16GeU,
        Op::I8x16Abs => Instruction::I8x16Abs,
        Op::I8x16Neg => Instruction::I8x16Neg,
        Op::I8x16AllTrue => Instruction::I8x16AllTrue,
        Op::I8x16Bitmask => Instruction::I8x16Bitmask,
        Op::I8x16Shl => Instruction::I8x16Shl,
        Op::I8x16ShrS => Instruction::I8x16ShrS,
        Op::I8x16ShrU => Instruction::I8x16ShrU,
        Op::I8x16Add => Instruction::I8x16Add,
        Op::I8x16AddSatS => Instruction::I8x16AddSatS,
        Op::I8x16AddSatU => Instruction::I8x16AddSatU,
        Op::I8x16Sub => Instruction::I8x16Sub,
        Op::I8x16SubSatS => Instruction::I8x16SubSatS,
        Op::I8x16SubSatU => Instruction::I8x16SubSatU,
        Op::I8x16MinS => Instruction::I8x16MinS,
        Op::I8x16MinU => Instruction::I8x16MinU,
        Op::I8x16MaxS => Instruction::I8x16MaxS,
        Op::I8x16MaxU => Instruction::I8x16MaxU,
        Op::I8x16AvgrU => Instruction::I8x16AvgrU,
        Op::I8x16Popcnt => Instruction::I8x16Popcnt,
        Op::I8x16NarrowI16x8S => Instruction::I8x16NarrowI16x8S,
        Op::I8x16NarrowI16x8U => Instruction::I8x16NarrowI16x8U,
        Op::I8x16Swizzle => Instruction::I8x16Swizzle,
        Op::I8x16Shuffle { lanes } => Instruction::I8x16Shuffle(*lanes),

        // ----- SIMD i16x8 arithmetic -----
        Op::I16x8Eq => Instruction::I16x8Eq,
        Op::I16x8Ne => Instruction::I16x8Ne,
        Op::I16x8LtS => Instruction::I16x8LtS,
        Op::I16x8LtU => Instruction::I16x8LtU,
        Op::I16x8GtS => Instruction::I16x8GtS,
        Op::I16x8GtU => Instruction::I16x8GtU,
        Op::I16x8LeS => Instruction::I16x8LeS,
        Op::I16x8LeU => Instruction::I16x8LeU,
        Op::I16x8GeS => Instruction::I16x8GeS,
        Op::I16x8GeU => Instruction::I16x8GeU,
        Op::I16x8Abs => Instruction::I16x8Abs,
        Op::I16x8Neg => Instruction::I16x8Neg,
        Op::I16x8AllTrue => Instruction::I16x8AllTrue,
        Op::I16x8Bitmask => Instruction::I16x8Bitmask,
        Op::I16x8Shl => Instruction::I16x8Shl,
        Op::I16x8ShrS => Instruction::I16x8ShrS,
        Op::I16x8ShrU => Instruction::I16x8ShrU,
        Op::I16x8Add => Instruction::I16x8Add,
        Op::I16x8AddSatS => Instruction::I16x8AddSatS,
        Op::I16x8AddSatU => Instruction::I16x8AddSatU,
        Op::I16x8Sub => Instruction::I16x8Sub,
        Op::I16x8SubSatS => Instruction::I16x8SubSatS,
        Op::I16x8SubSatU => Instruction::I16x8SubSatU,
        Op::I16x8Mul => Instruction::I16x8Mul,
        Op::I16x8MinS => Instruction::I16x8MinS,
        Op::I16x8MinU => Instruction::I16x8MinU,
        Op::I16x8MaxS => Instruction::I16x8MaxS,
        Op::I16x8MaxU => Instruction::I16x8MaxU,
        Op::I16x8AvgrU => Instruction::I16x8AvgrU,
        Op::I16x8NarrowI32x4S => Instruction::I16x8NarrowI32x4S,
        Op::I16x8NarrowI32x4U => Instruction::I16x8NarrowI32x4U,
        Op::I16x8ExtendLowI8x16S => Instruction::I16x8ExtendLowI8x16S,
        Op::I16x8ExtendHighI8x16S => Instruction::I16x8ExtendHighI8x16S,
        Op::I16x8ExtendLowI8x16U => Instruction::I16x8ExtendLowI8x16U,
        Op::I16x8ExtendHighI8x16U => Instruction::I16x8ExtendHighI8x16U,
        Op::I16x8ExtMulLowI8x16S => Instruction::I16x8ExtMulLowI8x16S,
        Op::I16x8ExtMulHighI8x16S => Instruction::I16x8ExtMulHighI8x16S,
        Op::I16x8ExtMulLowI8x16U => Instruction::I16x8ExtMulLowI8x16U,
        Op::I16x8ExtMulHighI8x16U => Instruction::I16x8ExtMulHighI8x16U,
        Op::I16x8Q15MulrSatS => Instruction::I16x8Q15MulrSatS,
        Op::I16x8ExtAddPairwiseI8x16S => Instruction::I16x8ExtAddPairwiseI8x16S,
        Op::I16x8ExtAddPairwiseI8x16U => Instruction::I16x8ExtAddPairwiseI8x16U,

        // ----- SIMD i32x4 arithmetic -----
        Op::I32x4Eq => Instruction::I32x4Eq,
        Op::I32x4Ne => Instruction::I32x4Ne,
        Op::I32x4LtS => Instruction::I32x4LtS,
        Op::I32x4LtU => Instruction::I32x4LtU,
        Op::I32x4GtS => Instruction::I32x4GtS,
        Op::I32x4GtU => Instruction::I32x4GtU,
        Op::I32x4LeS => Instruction::I32x4LeS,
        Op::I32x4LeU => Instruction::I32x4LeU,
        Op::I32x4GeS => Instruction::I32x4GeS,
        Op::I32x4GeU => Instruction::I32x4GeU,
        Op::I32x4Abs => Instruction::I32x4Abs,
        Op::I32x4Neg => Instruction::I32x4Neg,
        Op::I32x4AllTrue => Instruction::I32x4AllTrue,
        Op::I32x4Bitmask => Instruction::I32x4Bitmask,
        Op::I32x4Shl => Instruction::I32x4Shl,
        Op::I32x4ShrS => Instruction::I32x4ShrS,
        Op::I32x4ShrU => Instruction::I32x4ShrU,
        Op::I32x4Add => Instruction::I32x4Add,
        Op::I32x4Sub => Instruction::I32x4Sub,
        Op::I32x4Mul => Instruction::I32x4Mul,
        Op::I32x4MinS => Instruction::I32x4MinS,
        Op::I32x4MinU => Instruction::I32x4MinU,
        Op::I32x4MaxS => Instruction::I32x4MaxS,
        Op::I32x4MaxU => Instruction::I32x4MaxU,
        Op::I32x4DotI16x8S => Instruction::I32x4DotI16x8S,
        Op::I32x4ExtendLowI16x8S => Instruction::I32x4ExtendLowI16x8S,
        Op::I32x4ExtendHighI16x8S => Instruction::I32x4ExtendHighI16x8S,
        Op::I32x4ExtendLowI16x8U => Instruction::I32x4ExtendLowI16x8U,
        Op::I32x4ExtendHighI16x8U => Instruction::I32x4ExtendHighI16x8U,
        Op::I32x4ExtMulLowI16x8S => Instruction::I32x4ExtMulLowI16x8S,
        Op::I32x4ExtMulHighI16x8S => Instruction::I32x4ExtMulHighI16x8S,
        Op::I32x4ExtMulLowI16x8U => Instruction::I32x4ExtMulLowI16x8U,
        Op::I32x4ExtMulHighI16x8U => Instruction::I32x4ExtMulHighI16x8U,
        Op::I32x4ExtAddPairwiseI16x8S => Instruction::I32x4ExtAddPairwiseI16x8S,
        Op::I32x4ExtAddPairwiseI16x8U => Instruction::I32x4ExtAddPairwiseI16x8U,
        Op::I32x4TruncSatF32x4S => Instruction::I32x4TruncSatF32x4S,
        Op::I32x4TruncSatF32x4U => Instruction::I32x4TruncSatF32x4U,
        Op::I32x4TruncSatF64x2SZero => Instruction::I32x4TruncSatF64x2SZero,
        Op::I32x4TruncSatF64x2UZero => Instruction::I32x4TruncSatF64x2UZero,

        // ----- SIMD i64x2 arithmetic -----
        Op::I64x2Eq => Instruction::I64x2Eq,
        Op::I64x2Ne => Instruction::I64x2Ne,
        Op::I64x2LtS => Instruction::I64x2LtS,
        Op::I64x2GtS => Instruction::I64x2GtS,
        Op::I64x2LeS => Instruction::I64x2LeS,
        Op::I64x2GeS => Instruction::I64x2GeS,
        Op::I64x2Abs => Instruction::I64x2Abs,
        Op::I64x2Neg => Instruction::I64x2Neg,
        Op::I64x2AllTrue => Instruction::I64x2AllTrue,
        Op::I64x2Bitmask => Instruction::I64x2Bitmask,
        Op::I64x2Shl => Instruction::I64x2Shl,
        Op::I64x2ShrS => Instruction::I64x2ShrS,
        Op::I64x2ShrU => Instruction::I64x2ShrU,
        Op::I64x2Add => Instruction::I64x2Add,
        Op::I64x2Sub => Instruction::I64x2Sub,
        Op::I64x2Mul => Instruction::I64x2Mul,
        Op::I64x2ExtendLowI32x4S => Instruction::I64x2ExtendLowI32x4S,
        Op::I64x2ExtendHighI32x4S => Instruction::I64x2ExtendHighI32x4S,
        Op::I64x2ExtendLowI32x4U => Instruction::I64x2ExtendLowI32x4U,
        Op::I64x2ExtendHighI32x4U => Instruction::I64x2ExtendHighI32x4U,
        Op::I64x2ExtMulLowI32x4S => Instruction::I64x2ExtMulLowI32x4S,
        Op::I64x2ExtMulHighI32x4S => Instruction::I64x2ExtMulHighI32x4S,
        Op::I64x2ExtMulLowI32x4U => Instruction::I64x2ExtMulLowI32x4U,
        Op::I64x2ExtMulHighI32x4U => Instruction::I64x2ExtMulHighI32x4U,

        // ----- SIMD f32x4 arithmetic -----
        Op::F32x4Eq => Instruction::F32x4Eq,
        Op::F32x4Ne => Instruction::F32x4Ne,
        Op::F32x4Lt => Instruction::F32x4Lt,
        Op::F32x4Gt => Instruction::F32x4Gt,
        Op::F32x4Le => Instruction::F32x4Le,
        Op::F32x4Ge => Instruction::F32x4Ge,
        Op::F32x4Abs => Instruction::F32x4Abs,
        Op::F32x4Neg => Instruction::F32x4Neg,
        Op::F32x4Sqrt => Instruction::F32x4Sqrt,
        Op::F32x4Add => Instruction::F32x4Add,
        Op::F32x4Sub => Instruction::F32x4Sub,
        Op::F32x4Mul => Instruction::F32x4Mul,
        Op::F32x4Div => Instruction::F32x4Div,
        Op::F32x4Min => Instruction::F32x4Min,
        Op::F32x4Max => Instruction::F32x4Max,
        Op::F32x4PMin => Instruction::F32x4PMin,
        Op::F32x4PMax => Instruction::F32x4PMax,
        Op::F32x4Ceil => Instruction::F32x4Ceil,
        Op::F32x4Floor => Instruction::F32x4Floor,
        Op::F32x4Trunc => Instruction::F32x4Trunc,
        Op::F32x4Nearest => Instruction::F32x4Nearest,
        Op::F32x4ConvertI32x4S => Instruction::F32x4ConvertI32x4S,
        Op::F32x4ConvertI32x4U => Instruction::F32x4ConvertI32x4U,
        Op::F32x4DemoteF64x2Zero => Instruction::F32x4DemoteF64x2Zero,

        // ----- SIMD f64x2 arithmetic -----
        Op::F64x2Eq => Instruction::F64x2Eq,
        Op::F64x2Ne => Instruction::F64x2Ne,
        Op::F64x2Lt => Instruction::F64x2Lt,
        Op::F64x2Gt => Instruction::F64x2Gt,
        Op::F64x2Le => Instruction::F64x2Le,
        Op::F64x2Ge => Instruction::F64x2Ge,
        Op::F64x2Abs => Instruction::F64x2Abs,
        Op::F64x2Neg => Instruction::F64x2Neg,
        Op::F64x2Sqrt => Instruction::F64x2Sqrt,
        Op::F64x2Add => Instruction::F64x2Add,
        Op::F64x2Sub => Instruction::F64x2Sub,
        Op::F64x2Mul => Instruction::F64x2Mul,
        Op::F64x2Div => Instruction::F64x2Div,
        Op::F64x2Min => Instruction::F64x2Min,
        Op::F64x2Max => Instruction::F64x2Max,
        Op::F64x2PMin => Instruction::F64x2PMin,
        Op::F64x2PMax => Instruction::F64x2PMax,
        Op::F64x2Ceil => Instruction::F64x2Ceil,
        Op::F64x2Floor => Instruction::F64x2Floor,
        Op::F64x2Trunc => Instruction::F64x2Trunc,
        Op::F64x2Nearest => Instruction::F64x2Nearest,
        Op::F64x2ConvertLowI32x4S => Instruction::F64x2ConvertLowI32x4S,
        Op::F64x2ConvertLowI32x4U => Instruction::F64x2ConvertLowI32x4U,
        Op::F64x2PromoteLowF32x4 => Instruction::F64x2PromoteLowF32x4,

        // ----- SIMD V128 load/store variants -----
        Op::V128Load8Splat { memarg } => Instruction::V128Load8Splat(convert_memarg(memarg)),
        Op::V128Load16Splat { memarg } => Instruction::V128Load16Splat(convert_memarg(memarg)),
        Op::V128Load32Splat { memarg } => Instruction::V128Load32Splat(convert_memarg(memarg)),
        Op::V128Load64Splat { memarg } => Instruction::V128Load64Splat(convert_memarg(memarg)),
        Op::V128Load32Zero { memarg } => Instruction::V128Load32Zero(convert_memarg(memarg)),
        Op::V128Load64Zero { memarg } => Instruction::V128Load64Zero(convert_memarg(memarg)),
        Op::V128Load8x8S { memarg } => Instruction::V128Load8x8S(convert_memarg(memarg)),
        Op::V128Load8x8U { memarg } => Instruction::V128Load8x8U(convert_memarg(memarg)),
        Op::V128Load16x4S { memarg } => Instruction::V128Load16x4S(convert_memarg(memarg)),
        Op::V128Load16x4U { memarg } => Instruction::V128Load16x4U(convert_memarg(memarg)),
        Op::V128Load32x2S { memarg } => Instruction::V128Load32x2S(convert_memarg(memarg)),
        Op::V128Load32x2U { memarg } => Instruction::V128Load32x2U(convert_memarg(memarg)),
        Op::V128Load8Lane { memarg, lane } => Instruction::V128Load8Lane {
            memarg: convert_memarg(memarg),
            lane: *lane,
        },
        Op::V128Load16Lane { memarg, lane } => Instruction::V128Load16Lane {
            memarg: convert_memarg(memarg),
            lane: *lane,
        },
        Op::V128Load32Lane { memarg, lane } => Instruction::V128Load32Lane {
            memarg: convert_memarg(memarg),
            lane: *lane,
        },
        Op::V128Load64Lane { memarg, lane } => Instruction::V128Load64Lane {
            memarg: convert_memarg(memarg),
            lane: *lane,
        },
        Op::V128Store8Lane { memarg, lane } => Instruction::V128Store8Lane {
            memarg: convert_memarg(memarg),
            lane: *lane,
        },
        Op::V128Store16Lane { memarg, lane } => Instruction::V128Store16Lane {
            memarg: convert_memarg(memarg),
            lane: *lane,
        },
        Op::V128Store32Lane { memarg, lane } => Instruction::V128Store32Lane {
            memarg: convert_memarg(memarg),
            lane: *lane,
        },
        Op::V128Store64Lane { memarg, lane } => Instruction::V128Store64Lane {
            memarg: convert_memarg(memarg),
            lane: *lane,
        },

        // ----- Catch-all for any unhandled operator -----
        // The Operator enum is #[non_exhaustive] with 627+ variants.
        // Operators we haven't explicitly mapped will trap at runtime if hit.
        _ => Instruction::Unreachable,
    }
}

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

/// Convert wasmparser ValType to wasm-encoder ValType.
fn convert_val_type(vt: wasmparser::ValType) -> ValType {
    match vt {
        wasmparser::ValType::I32 => ValType::I32,
        wasmparser::ValType::I64 => ValType::I64,
        wasmparser::ValType::F32 => ValType::F32,
        wasmparser::ValType::F64 => ValType::F64,
        wasmparser::ValType::V128 => ValType::V128,
        wasmparser::ValType::Ref(rt) => {
            let ht = convert_heap_type(&rt.heap_type());
            ValType::Ref(wasm_encoder::RefType {
                nullable: rt.is_nullable(),
                heap_type: ht,
            })
        }
    }
}

/// Convert wasmparser HeapType to wasm-encoder HeapType.
fn convert_heap_type(ht: &wasmparser::HeapType) -> wasm_encoder::HeapType {
    use wasmparser::AbstractHeapType;
    match ht {
        wasmparser::HeapType::Abstract { shared, ty } => {
            let enc_ty = match ty {
                AbstractHeapType::Func => wasm_encoder::AbstractHeapType::Func,
                AbstractHeapType::Extern => wasm_encoder::AbstractHeapType::Extern,
                AbstractHeapType::Any => wasm_encoder::AbstractHeapType::Any,
                AbstractHeapType::None => wasm_encoder::AbstractHeapType::None,
                AbstractHeapType::NoExtern => wasm_encoder::AbstractHeapType::NoExtern,
                AbstractHeapType::NoFunc => wasm_encoder::AbstractHeapType::NoFunc,
                AbstractHeapType::Eq => wasm_encoder::AbstractHeapType::Eq,
                AbstractHeapType::Struct => wasm_encoder::AbstractHeapType::Struct,
                AbstractHeapType::Array => wasm_encoder::AbstractHeapType::Array,
                AbstractHeapType::I31 => wasm_encoder::AbstractHeapType::I31,
                AbstractHeapType::Exn => wasm_encoder::AbstractHeapType::Exn,
                AbstractHeapType::NoExn => wasm_encoder::AbstractHeapType::NoExn,
                _ => wasm_encoder::AbstractHeapType::Func,
            };
            wasm_encoder::HeapType::Abstract {
                shared: *shared,
                ty: enc_ty,
            }
        }
        wasmparser::HeapType::Concrete(index) => {
            wasm_encoder::HeapType::Concrete(index.as_module_index().unwrap_or(0))
        }
        wasmparser::HeapType::Exact(index) => {
            wasm_encoder::HeapType::Concrete(index.as_module_index().unwrap_or(0))
        }
    }
}

/// Convert wasmparser BlockType to wasm-encoder BlockType, remapping type indices.
fn convert_block_type(
    bt: &wasmparser::BlockType,
    type_remap: &BTreeMap<u32, u32>,
) -> wasm_encoder::BlockType {
    match bt {
        wasmparser::BlockType::Empty => wasm_encoder::BlockType::Empty,
        wasmparser::BlockType::Type(vt) => wasm_encoder::BlockType::Result(convert_val_type(*vt)),
        wasmparser::BlockType::FuncType(idx) => {
            let merged = type_remap.get(idx).copied().unwrap_or(*idx);
            wasm_encoder::BlockType::FunctionType(merged)
        }
    }
}

/// Convert wasmparser MemArg to wasm-encoder MemArg.
fn convert_memarg(ma: &wasmparser::MemArg) -> EncMemArg {
    EncMemArg {
        offset: ma.offset,
        align: ma.align as u32,
        memory_index: ma.memory,
    }
}

/// Parse a wasmparser ConstExpr into a simple GlobalInit.
fn parse_const_expr(expr: &wasmparser::ConstExpr) -> GlobalInit {
    let mut reader = expr.get_operators_reader();
    while let Ok(op) = reader.read() {
        match op {
            wasmparser::Operator::I32Const { value } => return GlobalInit::I32(value),
            wasmparser::Operator::I64Const { value } => return GlobalInit::I64(value),
            wasmparser::Operator::F32Const { value } => return GlobalInit::F32(value.bits()),
            wasmparser::Operator::F64Const { value } => return GlobalInit::F64(value.bits()),
            wasmparser::Operator::GlobalGet { global_index } => {
                return GlobalInit::GlobalGet(global_index)
            }
            wasmparser::Operator::End => break,
            _ => {}
        }
    }
    GlobalInit::I32(0)
}

/// Encode a GlobalInit as a wasm-encoder ConstExpr, remapping global references.
fn encode_global_init(init: &GlobalInit, global_remap: &BTreeMap<u32, u32>) -> ConstExpr {
    match init {
        GlobalInit::I32(v) => ConstExpr::i32_const(*v),
        GlobalInit::I64(v) => ConstExpr::i64_const(*v),
        GlobalInit::F32(bits) => ConstExpr::f32_const(wasm_encoder::Ieee32::new(*bits)),
        GlobalInit::F64(bits) => ConstExpr::f64_const(wasm_encoder::Ieee64::new(*bits)),
        GlobalInit::GlobalGet(idx) => {
            let merged = global_remap.get(idx).copied().unwrap_or(*idx);
            ConstExpr::global_get(merged)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid WASM module with wasm-encoder for testing.
    fn make_minimal_module(
        imports: &[(&str, &str, &[ValType], &[ValType])],
        num_funcs: usize,
        exports: &[(&str, u32)],
        memory: bool,
    ) -> Vec<u8> {
        let mut module = Module::new();

        let mut type_section = TypeSection::new();
        for (_, _, params, results) in imports {
            type_section
                .ty()
                .function(params.iter().copied(), results.iter().copied());
        }
        for _ in 0..num_funcs {
            type_section
                .ty()
                .function(std::iter::empty::<ValType>(), std::iter::empty::<ValType>());
        }
        module.section(&type_section);

        let mut import_section = ImportSection::new();
        for (i, (ns, name, _, _)) in imports.iter().enumerate() {
            import_section.import(ns, name, EntityType::Function(i as u32));
        }
        module.section(&import_section);

        let mut func_section = FunctionSection::new();
        for i in 0..num_funcs {
            func_section.function((imports.len() + i) as u32);
        }
        module.section(&func_section);

        if memory {
            let mut mem_section = MemorySection::new();
            mem_section.memory(EncMemoryType {
                minimum: 1,
                maximum: None,
                memory64: false,
                shared: false,
                page_size_log2: None,
            });
            module.section(&mem_section);
        }

        let mut export_section = ExportSection::new();
        for (name, idx) in exports {
            export_section.export(name, ExportKind::Func, *idx);
        }
        if memory {
            export_section.export("memory", ExportKind::Memory, 0);
        }
        module.section(&export_section);

        let mut code_section = CodeSection::new();
        for _ in 0..num_funcs {
            let mut func = Function::new(std::iter::empty::<(u32, ValType)>());
            func.instruction(&Instruction::End);
            code_section.function(&func);
        }
        module.section(&code_section);

        module.finish()
    }

    #[test]
    fn test_link_resolves_imports() {
        // Runtime: 1 WASI import, 1 internal function exported as "haxe_trace".
        let runtime = make_minimal_module(
            &[(
                "wasi_snapshot_preview1",
                "fd_write",
                &[ValType::I32, ValType::I32, ValType::I32, ValType::I32],
                &[ValType::I32],
            )],
            1,
            &[("haxe_trace", 1)], // func 0 = import, func 1 = internal
            true,
        );

        // User: 1 import from "rayzor" namespace, 1 internal function.
        let user = make_minimal_module(
            &[("rayzor", "haxe_trace", &[], &[])],
            1,
            &[("_start", 1)], // func 0 = import, func 1 = internal
            true,
        );

        let result = WasmLinker::link(&user, &runtime);
        assert!(result.is_ok(), "link failed: {:?}", result.err());

        let merged = result.unwrap();
        let parsed = ParsedModule::parse(&merged, "merged");
        assert!(parsed.is_ok(), "merged parse failed: {:?}", parsed.err());
        let parsed = parsed.unwrap();

        // Should have 1 WASI import (fd_write), not the rayzor import.
        assert_eq!(parsed.func_imports.len(), 1);
        assert_eq!(parsed.func_imports[0].module, "wasi_snapshot_preview1");
        assert_eq!(parsed.func_imports[0].name, "fd_write");

        // Should export _start.
        assert!(parsed.export_func_map.contains_key("_start"));
    }

    #[test]
    fn test_link_generates_stubs_for_unresolved() {
        // Runtime: no exports.
        let runtime = make_minimal_module(&[], 1, &[], true);

        // User: 1 import from "rayzor" that won't be resolved.
        let user = make_minimal_module(
            &[("rayzor", "nonexistent_func", &[], &[])],
            1,
            &[("_start", 1)],
            true,
        );

        let result = WasmLinker::link(&user, &runtime);
        assert!(result.is_ok(), "link failed: {:?}", result.err());

        let merged = result.unwrap();
        let parsed = ParsedModule::parse(&merged, "merged").unwrap();

        // No imports should remain (runtime had none, user's rayzor import was stubbed).
        assert_eq!(parsed.func_imports.len(), 0);

        // Should have 3 internal functions: 1 runtime + 1 stub + 1 user.
        assert_eq!(parsed.functions.len(), 3);
    }

    #[test]
    fn test_type_deduplication() {
        let mut linker = LinkerCtx::new();
        let sig1 = FuncSig {
            params: vec![ValType::I32],
            results: vec![ValType::I32],
        };
        let sig2 = FuncSig {
            params: vec![ValType::I32],
            results: vec![ValType::I32],
        };
        let sig3 = FuncSig {
            params: vec![ValType::I64],
            results: vec![],
        };

        let idx1 = linker.intern_type(&sig1);
        let idx2 = linker.intern_type(&sig2);
        let idx3 = linker.intern_type(&sig3);

        assert_eq!(idx1, idx2, "identical types should be deduplicated");
        assert_ne!(idx1, idx3, "different types should have different indices");
        assert_eq!(linker.merged_types.len(), 2);
    }

    #[test]
    fn test_call_index_rewriting() {
        // Build a user module where an internal function calls an imported function.
        let mut module = Module::new();

        let mut type_section = TypeSection::new();
        type_section
            .ty()
            .function(std::iter::empty::<ValType>(), std::iter::empty::<ValType>());
        module.section(&type_section);

        let mut import_section = ImportSection::new();
        import_section.import("rayzor", "do_thing", EntityType::Function(0));
        module.section(&import_section);

        let mut func_section = FunctionSection::new();
        func_section.function(0);
        module.section(&func_section);

        let mut mem_section = MemorySection::new();
        mem_section.memory(EncMemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mem_section);

        let mut export_section = ExportSection::new();
        export_section.export("_start", ExportKind::Func, 1);
        module.section(&export_section);

        let mut code_section = CodeSection::new();
        let mut func = Function::new(std::iter::empty::<(u32, ValType)>());
        func.instruction(&Instruction::Call(0)); // call do_thing (import idx 0)
        func.instruction(&Instruction::End);
        code_section.function(&func);
        module.section(&code_section);

        let user_wasm = module.finish();

        // Runtime: exports "do_thing" as func 0 (no imports, 1 internal).
        let runtime = make_minimal_module(&[], 1, &[("do_thing", 0)], true);

        let result = WasmLinker::link(&user_wasm, &runtime);
        assert!(result.is_ok(), "link failed: {:?}", result.err());

        let merged = result.unwrap();
        let parsed = ParsedModule::parse(&merged, "merged").unwrap();

        // No imports should remain.
        assert_eq!(parsed.func_imports.len(), 0);
        // 2 functions: runtime's do_thing + user's _start.
        assert_eq!(parsed.functions.len(), 2);
    }

    #[test]
    fn test_global_merging() {
        // Build runtime with 1 global (stack pointer).
        let mut rt_module = Module::new();
        let mut ts = TypeSection::new();
        ts.ty()
            .function(std::iter::empty::<ValType>(), std::iter::empty::<ValType>());
        rt_module.section(&ts);
        let mut fs = FunctionSection::new();
        fs.function(0);
        rt_module.section(&fs);
        let mut ms = MemorySection::new();
        ms.memory(EncMemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        rt_module.section(&ms);
        let mut gs = GlobalSection::new();
        gs.global(
            EncGlobalType {
                val_type: ValType::I32,
                mutable: true,
                shared: false,
            },
            &ConstExpr::i32_const(65536),
        );
        rt_module.section(&gs);
        let mut es = ExportSection::new();
        es.export("memory", ExportKind::Memory, 0);
        rt_module.section(&es);
        let mut cs = CodeSection::new();
        let mut f = Function::new(std::iter::empty::<(u32, ValType)>());
        f.instruction(&Instruction::End);
        cs.function(&f);
        rt_module.section(&cs);
        let runtime_wasm = rt_module.finish();

        // Build user with 1 global.
        let mut user_module = Module::new();
        let mut ts2 = TypeSection::new();
        ts2.ty()
            .function(std::iter::empty::<ValType>(), std::iter::empty::<ValType>());
        user_module.section(&ts2);
        let mut fs2 = FunctionSection::new();
        fs2.function(0);
        user_module.section(&fs2);
        let mut ms2 = MemorySection::new();
        ms2.memory(EncMemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        user_module.section(&ms2);
        let mut gs2 = GlobalSection::new();
        gs2.global(
            EncGlobalType {
                val_type: ValType::I64,
                mutable: true,
                shared: false,
            },
            &ConstExpr::i64_const(42),
        );
        user_module.section(&gs2);
        let mut es2 = ExportSection::new();
        es2.export("_start", ExportKind::Func, 0);
        user_module.section(&es2);
        let mut cs2 = CodeSection::new();
        let mut f2 = Function::new(std::iter::empty::<(u32, ValType)>());
        f2.instruction(&Instruction::End);
        cs2.function(&f2);
        user_module.section(&cs2);
        let user_wasm = user_module.finish();

        let result = WasmLinker::link(&user_wasm, &runtime_wasm);
        assert!(result.is_ok(), "link failed: {:?}", result.err());

        let merged = result.unwrap();
        let parsed = ParsedModule::parse(&merged, "merged").unwrap();

        // Should have 2 globals: runtime's stack pointer + user's.
        assert_eq!(parsed.globals.len(), 2);
    }
}
