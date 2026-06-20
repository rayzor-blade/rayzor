//! WebAssembly code generation backend (Phase 1).
//!
//! Translates MIR (SSA-form IR) into a WASM binary using `wasm-encoder`.
//! This is a Phase 1 backend: correctness over performance. Control flow
//! is lowered via a simple br_table dispatch loop, and phi nodes are
//! resolved with explicit WASM locals at branch sites.
//!
//! Pipeline: MIR (IrModule) -> wasm-encoder sections -> `.wasm` binary bytes
//!
//! # Architecture
//!
//! MIR (IrModule) -> WasmBackend::compile() -> Vec<u8> (.wasm binary)
//!
//! The backend handles:
//! - MIR instruction -> WASM instruction mapping
//! - CFG -> structured control flow (block/loop/br_table dispatch)
//! - Phi nodes -> explicit WASM locals (resolved at branch sites)
//! - Runtime imports for stdlib functions
//! - Linear memory management (shadow stack + heap via imports)
//! - String constant pool in the data section

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, GlobalSection, GlobalType, Ieee32, Ieee64, ImportSection,
    Instruction, MemArg, MemorySection, MemoryType, Module, TypeSection, ValType,
};

use crate::ir::blocks::{IrBasicBlock, IrBlockId, IrTerminator};
use crate::ir::functions::{IrFunction, IrFunctionId, IrFunctionSignature};
use crate::ir::instructions::{
    AtomicRmwOp, BinaryOp, CompareOp, IrInstruction, UnaryOp, VectorMinMaxKind, VectorUnaryOpKind,
};
use crate::ir::modules::{IrGlobalId, IrModule};
use crate::ir::{IrId, IrType, IrValue};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Initial linear memory: 256 pages = 16 MiB.
const INITIAL_MEMORY_PAGES: u64 = 256;

/// Stack pointer starts near the top of initial memory (minus a guard page).
const STACK_POINTER_INIT: i32 = (INITIAL_MEMORY_PAGES as i32) * 65536 - 4096;

/// Data section starts at this offset (leave room for null-guard page).
const DATA_SECTION_BASE: u32 = 4096;

/// Global index 0 is always the mutable `__stack_pointer`.
const STACK_PTR_GLOBAL: u32 = 0;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// WebAssembly compilation backend.
pub struct WasmBackend;

impl WasmBackend {
    /// Compile MIR modules to a WASM binary.
    ///
    /// `entry_function` -- if provided, the named function is exported as `_start`.
    /// Returns raw `.wasm` bytes on success.
    pub fn compile(modules: &[&IrModule], entry_function: Option<&str>) -> Result<Vec<u8>, String> {
        Self::compile_with_method_map(modules, entry_function, &std::collections::BTreeMap::new())
    }

    /// Compile with a qualified method name → native name map for stub resolution.
    /// Maps e.g. "rayzor.gpu.Surface.getFormat" → "rayzor_gpu_gfx_surface_get_format"
    /// so the WASM backend can generate proper forwarder functions instead of unreachable stubs.
    pub fn compile_with_method_map(
        modules: &[&IrModule],
        entry_function: Option<&str>,
        qualified_method_map: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<u8>, String> {
        let mut ctx = CompileCtx::new();
        ctx.collect_imports(modules);
        ctx.collect_indirect_targets(modules);
        ctx.collect_functions(modules);
        // Build qualified_to_import from the method map
        for (qualified, native) in qualified_method_map {
            if let Some(&idx) = ctx.import_name_to_idx.get(native) {
                ctx.qualified_to_import.insert(qualified.clone(), idx);
            }
        }
        ctx.build_func_id_fallback(modules);
        // Pre-allocate table slots for all MakeClosure targets
        {
            use crate::ir::IrInstruction;
            for m in modules {
                for func in m.functions.values() {
                    for block in func.cfg.blocks.values() {
                        for inst in &block.instructions {
                            if let IrInstruction::MakeClosure { func_id, .. } = inst {
                                let fn_idx = ctx.ir_func_to_idx.get(func_id).copied().unwrap_or(0);
                                ctx.get_table_slot(fn_idx);
                            }
                        }
                    }
                }
            }
        }
        // Build per-module func_id→name maps for last-resort name resolution
        if ctx.all_module_funcs.is_empty() {
            for m in modules {
                let mut map = BTreeMap::new();
                for (id, f) in &m.functions {
                    map.insert(*id, f.name.clone());
                }
                for (_, f) in &m.extern_functions {
                    map.insert(f.id, f.name.clone());
                }
                ctx.all_module_funcs.push(map);
            }
        }
        ctx.collect_strings(modules);
        ctx.collect_globals(modules);
        for m in modules {
            for func in m.functions.values() {
                if func.wasm_export {
                    if let Some(ref qn) = func.qualified_name {
                        if let Some(dot) = qn.rfind('.') {
                            ctx.exported_classes.insert(qn[..dot].to_string());
                        }
                    }
                }
            }
        }
        ctx.encode_with_exports(modules, entry_function, &[])
    }

    /// Compile with extern function → JS module mappings from @:jsImport.
    pub fn compile_with_js_modules(
        modules: &[&IrModule],
        entry_function: Option<&str>,
        extra_exports: &[&str],
        extern_js_modules: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<u8>, String> {
        let mut ctx = CompileCtx::new();
        // Pre-populate js_import_modules from extern class @:jsImport mappings
        for (func_name, module_name) in extern_js_modules {
            ctx.js_import_modules
                .insert(func_name.clone(), (module_name.clone(), func_name.clone()));
        }
        ctx.collect_imports(modules);
        ctx.collect_indirect_targets(modules);
        ctx.collect_functions(modules);
        // Build fallback map: scan ALL functions for CallDirect targets not in ir_func_to_idx,
        // resolve them by name from extern_functions/functions across all modules.
        ctx.build_func_id_fallback(modules);
        // Pre-allocate table slots for all MakeClosure targets
        {
            use crate::ir::IrInstruction;
            for m in modules {
                for func in m.functions.values() {
                    for block in func.cfg.blocks.values() {
                        for inst in &block.instructions {
                            if let IrInstruction::MakeClosure { func_id, .. } = inst {
                                let fn_idx = ctx.ir_func_to_idx.get(func_id).copied().unwrap_or(0);
                                ctx.get_table_slot(fn_idx);
                            }
                        }
                    }
                }
            }
        }
        // Build per-module func_id→name maps for last-resort name resolution
        if ctx.all_module_funcs.is_empty() {
            for m in modules {
                let mut map = BTreeMap::new();
                for (id, f) in &m.functions {
                    map.insert(*id, f.name.clone());
                }
                for (_, f) in &m.extern_functions {
                    map.insert(f.id, f.name.clone());
                }
                ctx.all_module_funcs.push(map);
            }
        }
        ctx.collect_strings(modules);
        ctx.collect_globals(modules);
        for m in modules {
            for func in m.functions.values() {
                if func.wasm_export {
                    if let Some(ref qn) = func.qualified_name {
                        if let Some(dot) = qn.rfind('.') {
                            ctx.exported_classes.insert(qn[..dot].to_string());
                        }
                    }
                }
            }
        }
        ctx.encode_with_exports(modules, entry_function, extra_exports)
    }

    /// Compile with additional function exports (for JS interop).
    pub fn compile_with_exports(
        modules: &[&IrModule],
        entry_function: Option<&str>,
        extra_exports: &[&str],
    ) -> Result<Vec<u8>, String> {
        Self::compile_with_js_modules(
            modules,
            entry_function,
            extra_exports,
            &std::collections::BTreeMap::new(),
        )
    }
}

// ---------------------------------------------------------------------------
// Compile context -- accumulates state across all modules
// ---------------------------------------------------------------------------

/// Tracks an imported function (from the "rayzor" runtime namespace).
#[derive(Debug, Clone)]
struct ImportedFunc {
    name: String,
    type_idx: u32,
}

/// Tracks an internal (user-defined / MIR) function.
#[derive(Debug, Clone)]
struct InternalFunc {
    ir_id: IrFunctionId,
    module_idx: usize,
    type_idx: u32,
    func_idx: u32,
}

/// A string constant stored in the data section.
struct DataString {
    offset: u32,
    bytes: Vec<u8>,
}

struct CompileCtx {
    // ----- Type section -----
    type_map: BTreeMap<(Vec<ValType>, Vec<ValType>), u32>,
    types: Vec<(Vec<ValType>, Vec<ValType>)>,

    // ----- Import section -----
    imports: Vec<ImportedFunc>,
    import_name_to_idx: BTreeMap<String, u32>,

    // ----- Internal functions -----
    internals: Vec<InternalFunc>,
    /// IrFunctionId -> absolute WASM function index.
    ir_func_to_idx: BTreeMap<IrFunctionId, u32>,
    /// Name -> absolute func_idx for entry-point lookup.
    func_name_to_idx: BTreeMap<String, u32>,

    next_func_idx: u32,

    // ----- String pool / data section -----
    string_entries: Vec<DataString>,
    string_offsets: BTreeMap<String, u32>,
    data_offset: u32,

    // ----- Globals -----
    ir_global_to_idx: BTreeMap<IrGlobalId, u32>,
    /// User globals (excluding __stack_pointer which is always index 0).
    user_globals: Vec<(IrGlobalId, ValType, i64)>,

    // ----- Function return types -----
    /// IrFunctionId -> WASM return type. Used for CallDirect dest type inference.
    func_return_types: BTreeMap<IrFunctionId, ValType>,
    /// IrFunctionId -> WASM parameter types. Used for CallDirect arg coercion.
    func_param_types: BTreeMap<IrFunctionId, Vec<ValType>>,
    /// Class names that have @:export (for constructor detection).
    exported_classes: std::collections::BTreeSet<String>,
    /// @:jsImport function name → (js_module, js_import_name).
    /// Used to emit WASM imports from named JS modules instead of the default "rayzor" module.
    js_import_modules: BTreeMap<String, (String, String)>,
    /// Per-module func_id → name maps for last-resort name-based resolution.
    all_module_funcs: Vec<BTreeMap<IrFunctionId, String>>,
    /// Fallback IrFunctionId → WASM func index for cross-module extern resolution.
    /// Populated after collect_imports + collect_functions to resolve IDs not in ir_func_to_idx.
    func_id_fallback: BTreeMap<IrFunctionId, u32>,
    /// Function indices that need indirect function table entries (for closures/call_indirect).
    /// Maps WASM func index → table slot index.
    table_entries: BTreeMap<u32, u32>,
    /// Next available table slot.
    next_table_slot: u32,
    /// MIR functions that are used as closure/function-reference targets.
    indirect_target_functions: BTreeSet<IrFunctionId>,
    /// MIR functions whose WASM signature was widened with a hidden env pointer.
    functions_with_env: BTreeSet<IrFunctionId>,
    /// Qualified name → import index. Maps "rayzor.gpu.Surface.getFormat" → import idx for
    /// "rayzor_gpu_gfx_surface_get_format". Built from extern function qualified_name fields.
    qualified_to_import: BTreeMap<String, u32>,
    // ----- WASM memory reclamation (Phase 1: transient string frees) -----
    /// Import index of `haxe_string_free` (frees a heap HaxeString's data buffer),
    /// when the string runtime is linked. `None` if strings aren't used by the module.
    string_free_idx: Option<u32>,
    /// Import index of `free` (linker-aliased to the size-prefix-header
    /// `rayzor_obj_free`). Used to lower `IrInstruction::Free { ptr }` to a real
    /// reclamation — the wasm analogue of native's `call free(ptr)`. `None` if
    /// the runtime allocator isn't linked.
    free_idx: Option<u32>,
    /// Master kill-switch for the wasm reclamation pass. `RAYZOR_WASM_FREE=0` disables
    /// all emitted frees (for A/B + bisection); defaults to enabled.
    wasm_free_enabled: bool,
}

impl CompileCtx {
    fn new() -> Self {
        Self {
            type_map: BTreeMap::new(),
            types: Vec::new(),
            imports: Vec::new(),
            import_name_to_idx: BTreeMap::new(),
            internals: Vec::new(),
            ir_func_to_idx: BTreeMap::new(),
            func_name_to_idx: BTreeMap::new(),
            next_func_idx: 0,
            string_entries: Vec::new(),
            string_offsets: BTreeMap::new(),
            data_offset: DATA_SECTION_BASE,
            ir_global_to_idx: BTreeMap::new(),
            user_globals: Vec::new(),
            func_return_types: BTreeMap::new(),
            func_param_types: BTreeMap::new(),
            exported_classes: std::collections::BTreeSet::new(),
            js_import_modules: BTreeMap::new(),
            all_module_funcs: Vec::new(),
            func_id_fallback: BTreeMap::new(),
            qualified_to_import: BTreeMap::new(),
            table_entries: BTreeMap::new(),
            next_table_slot: 1, // slot 0 is reserved (null)
            indirect_target_functions: BTreeSet::new(),
            functions_with_env: BTreeSet::new(),
            string_free_idx: None,
            free_idx: None,
            // Reclamation defaults ON; `RAYZOR_WASM_FREE=0` opts out for A/B + bisection.
            wasm_free_enabled: std::env::var("RAYZOR_WASM_FREE").as_deref() != Ok("0"),
        }
    }

    /// Intern a function type, returning its type-section index.
    fn intern_type(&mut self, params: Vec<ValType>, results: Vec<ValType>) -> u32 {
        let key = (params.clone(), results.clone());
        if let Some(&idx) = self.type_map.get(&key) {
            return idx;
        }
        let idx = self.types.len() as u32;
        self.types.push(key.clone());
        self.type_map.insert(key, idx);
        idx
    }

    fn sig_to_wasm(sig: &IrFunctionSignature) -> (Vec<ValType>, Vec<ValType>) {
        let params: Vec<ValType> = sig
            .parameters
            .iter()
            .map(|p| ir_type_to_wasm(&p.ty))
            .collect();
        let results = if matches!(sig.return_type, IrType::Void) {
            vec![]
        } else {
            vec![ir_type_to_wasm(&sig.return_type)]
        };
        (params, results)
    }

    fn sig_to_wasm_named(name: &str, sig: &IrFunctionSignature) -> (Vec<ValType>, Vec<ValType>) {
        match name {
            // Anonymous-object slots store raw Haxe Dynamic payload bits. Keep the
            // value lane as i64 so F64/String/handle payloads round-trip through
            // set/get without wasm32 Int narrowing.
            "rayzor_anon_set_field_by_index" => {
                (vec![ValType::I32, ValType::I32, ValType::I64], vec![])
            }
            "rayzor_anon_get_field_by_index" => {
                (vec![ValType::I32, ValType::I32], vec![ValType::I64])
            }
            "rayzor_anon_new" => (vec![ValType::I32, ValType::I32], vec![ValType::I32]),
            "rayzor_ensure_shape" => (vec![ValType::I32, ValType::I32], vec![]),
            "rayzor_anon_clone" | "rayzor_anon_copy" => (vec![ValType::I32], vec![ValType::I32]),
            "rayzor_anon_drop" => (vec![ValType::I32], vec![]),
            _ => Self::sig_to_wasm(sig),
        }
    }

    fn already_has_env_param(func: &IrFunction) -> bool {
        func.signature
            .parameters
            .first()
            .map_or(false, |p| p.name == "env")
    }

    fn needs_implicit_env(&self, func_id: &IrFunctionId, func: &IrFunction) -> bool {
        self.indirect_target_functions.contains(func_id)
            && !Self::already_has_env_param(func)
            && func.signature.calling_convention != crate::ir::CallingConvention::C
            && !func.cfg.blocks.is_empty()
    }

    fn wasm_sig_for_internal(
        &mut self,
        func_id: &IrFunctionId,
        func: &IrFunction,
    ) -> (Vec<ValType>, Vec<ValType>) {
        let (mut params, results) = Self::sig_to_wasm_named(&func.name, &func.signature);
        if self.needs_implicit_env(func_id, func) {
            params.insert(0, ValType::I32);
            self.functions_with_env.insert(*func_id);
        }
        (params, results)
    }

    fn collect_indirect_targets(&mut self, modules: &[&IrModule]) {
        use crate::ir::IrInstruction;

        for module in modules {
            for func in module.functions.values() {
                for block in func.cfg.blocks.values() {
                    for inst in &block.instructions {
                        match inst {
                            IrInstruction::FunctionRef { func_id, .. }
                            | IrInstruction::MakeClosure { func_id, .. } => {
                                self.indirect_target_functions.insert(*func_id);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 1a -- imports (extern functions that have NO internal body)
    // ------------------------------------------------------------------

    fn collect_imports(&mut self, modules: &[&IrModule]) {
        // Build set of IrFunctionIds that have actual code bodies (internal functions).
        let mut has_code: BTreeSet<IrFunctionId> = BTreeSet::new();
        for module in modules {
            for (func_id, func) in &module.functions {
                if !func.cfg.blocks.is_empty() {
                    has_code.insert(*func_id);
                }
            }
        }

        // Pre-scan: collect @:jsImport mappings
        for module in modules {
            for (_id, func) in &module.functions {
                if let Some((ref js_mod, ref js_name)) = func.js_import {
                    self.js_import_modules
                        .insert(func.name.clone(), (js_mod.clone(), js_name.clone()));
                }
            }
        }

        // ============================================================
        // DETERMINISTIC IMPORT RESOLUTION
        // ============================================================
        // Phase 1: Collect ALL unique import names from externs + empty-body functions.
        // Phase 2: Sort by name, assign WASM indices in sorted order.
        // Phase 3: Map ALL IrFunctionIds to their name's index.
        // This makes import ordering identical across builds regardless of
        // non-deterministic IrFunctionId assignment during compilation.
        // ============================================================

        // Phase 1: Collect unique import entries (name → signature) from all sources.
        // Use BTreeMap for deterministic iteration.
        let mut import_entries: BTreeMap<String, (Vec<ValType>, Vec<ValType>)> = BTreeMap::new();

        // From extern_functions — redirect bare names to qualified forms
        let mut bare_to_qualified: BTreeMap<String, String> = BTreeMap::new();
        for module in modules {
            for (_id, ext) in &module.extern_functions {
                if has_code.contains(&ext.id) {
                    continue;
                }
                let is_bare = !ext.name.contains('_') && !ext.name.contains('.');
                if is_bare {
                    let snake: String =
                        ext.name
                            .chars()
                            .enumerate()
                            .fold(String::new(), |mut s, (i, c)| {
                                if c.is_uppercase() && i > 0 {
                                    s.push('_');
                                }
                                s.push(c.to_ascii_lowercase());
                                s
                            });
                    // Known bare method names → synthesize qualified import
                    let qualified = match ext.name.as_str() {
                        "get" | "set" | "sub" | "blit" | "fill" | "compare" | "getInt16"
                        | "setInt16" | "getInt32" | "setInt32" | "getInt64" | "setInt64"
                        | "getFloat" | "setFloat" | "getDouble" | "setDouble" | "length"
                        | "alloc" | "ofString" | "toString" => {
                            Some(format!("haxe_bytes_{}", snake))
                        }
                        "lock" | "unlock" | "isLocked" | "tryLock" => {
                            Some(format!("rayzor_mutex_{}", snake))
                        }
                        _ => None,
                    };
                    if let Some(qname) = qualified {
                        bare_to_qualified.insert(ext.name.clone(), qname.clone());
                        import_entries
                            .entry(qname)
                            .or_insert_with(|| Self::sig_to_wasm_named(&ext.name, &ext.signature));
                        continue;
                    }
                }
                import_entries
                    .entry(ext.name.clone())
                    .or_insert_with(|| Self::sig_to_wasm_named(&ext.name, &ext.signature));
            }
        }

        // From empty-body functions in module.functions
        // bare_to_qualified is shared with the extern_functions phase above.
        for module in modules {
            for (_id, func) in &module.functions {
                if !func.cfg.blocks.is_empty() {
                    continue;
                }
                // Redirect bare-name stubs to qualified haxe_* imports.
                // "setInt32" → "haxe_bytes_set_int32" (either existing or synthesized).
                let is_bare = !func.name.contains('_') && !func.name.contains('.');
                if is_bare {
                    let snake: String =
                        func.name
                            .chars()
                            .enumerate()
                            .fold(String::new(), |mut s, (i, c)| {
                                if c.is_uppercase() && i > 0 {
                                    s.push('_');
                                }
                                s.push(c.to_ascii_lowercase());
                                s
                            });
                    let suffix = format!("_{}", snake);
                    // Check if a qualified import already exists (haxe_* or rayzor_*)
                    let existing_qualified = import_entries
                        .keys()
                        .find(|k| {
                            (k.starts_with("haxe_") || k.starts_with("rayzor_"))
                                && k.ends_with(&suffix)
                        })
                        .cloned();
                    if let Some(qname) = existing_qualified {
                        bare_to_qualified.insert(func.name.clone(), qname);
                        continue; // Will be mapped in Phase 3
                    }
                    // No qualified version exists — try to synthesize one from the
                    // function's qualified_name or a known class prefix.
                    if let Some(ref qn) = func.qualified_name {
                        // e.g., "rayzor.Bytes.setInt32" → "haxe_bytes_set_int32"
                        let qualified_import = qn.replace('.', "_").to_lowercase();
                        if qualified_import != func.name {
                            bare_to_qualified.insert(func.name.clone(), qualified_import.clone());
                            import_entries.entry(qualified_import).or_insert_with(|| {
                                Self::sig_to_wasm_named(&func.name, &func.signature)
                            });
                            continue;
                        }
                    }
                    // Last resort: use "haxe_bytes_" prefix for known Bytes method names
                    let qualified = match func.name.as_str() {
                        "get" | "set" | "sub" | "blit" | "fill" | "compare" | "getInt16"
                        | "setInt16" | "getInt32" | "setInt32" | "getInt64" | "setInt64"
                        | "getFloat" | "setFloat" | "getDouble" | "setDouble" | "length"
                        | "alloc" | "ofString" | "toString" => {
                            Some(format!("haxe_bytes_{}", snake))
                        }
                        "lock" | "unlock" | "isLocked" | "tryLock" => {
                            Some(format!("rayzor_mutex_{}", snake))
                        }
                        _ => None,
                    };
                    if let Some(qname) = qualified {
                        bare_to_qualified.insert(func.name.clone(), qname.clone());
                        import_entries.entry(qname).or_insert_with(|| {
                            Self::sig_to_wasm_named(&func.name, &func.signature)
                        });
                        continue;
                    }
                }
                import_entries
                    .entry(func.name.clone())
                    .or_insert_with(|| Self::sig_to_wasm_named(&func.name, &func.signature));
            }
        }

        // Reclamation (Phase 1): if the string runtime is linked (any haxe_string_*
        // import is present), force-import `haxe_string_free` so the backend can emit
        // frees for transient heap strings even though user code never calls it. Gated
        // on string-runtime presence so non-string modules don't gain an unresolvable
        // import. Signature: (i32) -> () — takes a HaxeString*, frees its data buffer.
        if self.wasm_free_enabled && import_entries.keys().any(|n| n.starts_with("haxe_string_")) {
            import_entries
                .entry("haxe_string_free".to_string())
                .or_insert_with(|| (vec![ValType::I32], vec![]));
        }

        // Reclamation (Phase 1): force-import `free` (linker-aliased to the
        // size-prefix-header rayzor_obj_free) so the backend can lower
        // IrInstruction::Free{ptr} to a real reclamation. Gated on the allocator
        // being linked (the module imports `malloc`). InsertFreePass emits Free
        // only for non-escaping, dominating, entry-block malloc-class objects, so
        // the dropped pointer is provably dead. Signature: (i32) -> ().
        if self.wasm_free_enabled && import_entries.keys().any(|n| n == "malloc") {
            import_entries
                .entry("free".to_string())
                .or_insert_with(|| (vec![ValType::I32], vec![]));
        }

        // Phase 2: Create imports in deterministic name order (BTreeMap iterates sorted).
        // Build name→index map.
        let mut name_to_idx: BTreeMap<String, u32> = BTreeMap::new();
        for (name, (params, results)) in &import_entries {
            let param_types = params.clone();
            let ret_type = if results.is_empty() {
                ValType::I32
            } else {
                results[0]
            };
            let type_idx = self.intern_type(params.clone(), results.clone());
            let func_idx = self.next_func_idx;
            self.next_func_idx += 1;
            self.imports.push(ImportedFunc {
                name: name.clone(),
                type_idx,
            });
            self.import_name_to_idx.insert(name.clone(), func_idx);
            name_to_idx.insert(name.clone(), func_idx);
        }

        // Cache the reclamation free-fn index for the FunctionLowerer.
        self.string_free_idx = self.import_name_to_idx.get("haxe_string_free").copied();
        self.free_idx = self.import_name_to_idx.get("free").copied();

        // Phase 3: Map ALL IrFunctionIds to their import index via name lookup.
        // This is the key step — IDs resolve by NAME, not by iteration order.
        for module in modules {
            // Map extern function IDs
            for (_id, ext) in &module.extern_functions {
                if has_code.contains(&ext.id) {
                    continue;
                }
                // Try direct name, then bare→qualified redirect
                let resolved_name = bare_to_qualified.get(&ext.name).unwrap_or(&ext.name);
                if let Some(&idx) = name_to_idx.get(resolved_name) {
                    self.ir_func_to_idx.entry(ext.id).or_insert(idx);
                    let (params, _) = Self::sig_to_wasm_named(&ext.name, &ext.signature);
                    let ret = if let Some((_, results)) = import_entries.get(resolved_name) {
                        if results.is_empty() {
                            ValType::I32
                        } else {
                            results[0]
                        }
                    } else {
                        ValType::I32
                    };
                    self.func_param_types.entry(ext.id).or_insert(params);
                    self.func_return_types.entry(ext.id).or_insert(ret);
                }
            }
            // Map empty-body function IDs
            for (func_id, func) in &module.functions {
                if !func.cfg.blocks.is_empty() {
                    continue;
                }
                if self.ir_func_to_idx.contains_key(func_id) {
                    continue;
                }
                // Direct name match
                if let Some(&idx) = name_to_idx.get(&func.name) {
                    self.ir_func_to_idx.insert(*func_id, idx);
                    continue;
                }
                // Bare→qualified redirect from Phase 1
                if let Some(qname) = bare_to_qualified.get(&func.name) {
                    if let Some(&idx) = name_to_idx.get(qname) {
                        self.ir_func_to_idx.insert(*func_id, idx);
                        continue;
                    }
                }
                // Bare-name redirect: "get" → "haxe_bytes_get" (fallback scan)
                if !func.name.contains('_') && !func.name.contains('.') {
                    let snake: String =
                        func.name
                            .chars()
                            .enumerate()
                            .fold(String::new(), |mut s, (i, c)| {
                                if c.is_uppercase() && i > 0 {
                                    s.push('_');
                                }
                                s.push(c.to_ascii_lowercase());
                                s
                            });
                    let suffix = format!("_{}", snake);
                    if let Some((_qname, &idx)) = name_to_idx
                        .iter()
                        .filter(|(k, _)| {
                            (k.starts_with("haxe_") || k.starts_with("rayzor_"))
                                && k.ends_with(&suffix)
                        })
                        .min_by_key(|(k, _)| k.len())
                    {
                        self.ir_func_to_idx.insert(*func_id, idx);
                        continue;
                    }
                }
            }
        }

        // Build qualified_to_import from function qualified_name fields
        for module in modules {
            for func in module.functions.values() {
                if let Some(ref qn) = func.qualified_name {
                    if !self.qualified_to_import.contains_key(qn) {
                        if let Some(&idx) = self.import_name_to_idx.get(&func.name) {
                            self.qualified_to_import.insert(qn.clone(), idx);
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 1b -- internal functions
    // ------------------------------------------------------------------

    fn collect_functions(&mut self, modules: &[&IrModule]) {
        // Sort functions by name for deterministic index assignment.
        // IrFunctionIds vary between builds, but names are stable.
        let mut all_funcs: Vec<(usize, IrFunctionId, &IrFunction)> = modules
            .iter()
            .enumerate()
            .flat_map(|(mod_idx, module)| {
                module
                    .functions
                    .iter()
                    .map(move |(fid, f)| (mod_idx, *fid, f))
            })
            .collect();
        all_funcs.sort_by(|a, b| a.2.name.cmp(&b.2.name).then(a.1.cmp(&b.1)));

        for &(mod_idx, func_id, func) in &all_funcs {
            let func_id = &func_id; // match old code's reference pattern
                                    // Skip functions already registered as imports (by ID or name).
                                    // Always register return type for CallDirect type inference
                                    // If this function maps to an import, use the import's signature for param types
                                    // (the import type is authoritative — it may have f64 for Float params while
                                    // the IrFunction might have Ptr/i32 from an older compilation context).
            let (param_vts, ret_vt) = if let Some(&idx) = self.ir_func_to_idx.get(func_id) {
                if (idx as usize) < self.imports.len() {
                    let imp = &self.imports[idx as usize];
                    self.types
                        .get(imp.type_idx as usize)
                        .map(|(p, r)| (p.clone(), r.first().copied().unwrap_or(ValType::I32)))
                        .unwrap_or_else(|| {
                            let p: Vec<ValType> = func
                                .signature
                                .parameters
                                .iter()
                                .map(|p| ir_type_to_wasm(&p.ty))
                                .collect();
                            (p, ir_type_to_wasm(&func.signature.return_type))
                        })
                } else {
                    let (p, r) = self.wasm_sig_for_internal(func_id, func);
                    (p, r.first().copied().unwrap_or(ValType::I32))
                }
            } else {
                let (p, r) = self.wasm_sig_for_internal(func_id, func);
                (p, r.first().copied().unwrap_or(ValType::I32))
            };
            self.func_return_types.insert(*func_id, ret_vt);
            self.func_param_types.insert(*func_id, param_vts);

            if self.ir_func_to_idx.contains_key(func_id) {
                let idx = *self.ir_func_to_idx.get(func_id).unwrap();
                self.func_name_to_idx.insert(func.name.clone(), idx);
                continue;
            }

            // @:jsImport functions become WASM imports from a named JS module
            if let Some((ref js_module, ref js_name)) = func.js_import {
                let (params, results) = Self::sig_to_wasm_named(&func.name, &func.signature);
                let type_idx = self.intern_type(params, results);
                let func_idx = self.next_func_idx;
                self.next_func_idx += 1;
                self.imports.push(ImportedFunc {
                    name: func.name.clone(),
                    type_idx,
                });
                // Store the JS module name for this import (used in import section encoding)
                self.js_import_modules
                    .insert(func.name.clone(), (js_module.clone(), js_name.clone()));
                self.import_name_to_idx.insert(func.name.clone(), func_idx);
                self.ir_func_to_idx.insert(*func_id, func_idx);
                self.func_name_to_idx.insert(func.name.clone(), func_idx);
                continue;
            }
            if let Some(&idx) = self.import_name_to_idx.get(&func.name) {
                self.ir_func_to_idx.insert(*func_id, idx);
                self.func_name_to_idx.insert(func.name.clone(), idx);
                continue;
            }
            // Map CallDirect targets in this function to import indices.
            // This doesn't remove the function — it just ensures the func_id→idx
            // mapping for any CallDirect targets is available.
            {
                use crate::ir::IrInstruction;
                for block in func.cfg.blocks.values() {
                    for inst in &block.instructions {
                        if let IrInstruction::CallDirect { func_id: fid, .. } = inst {
                            if !self.ir_func_to_idx.contains_key(fid) {
                                // Try to resolve the target by name across all modules
                                'resolve: for m2 in modules.iter() {
                                    if let Some(tf) = m2.extern_functions.get(fid) {
                                        if let Some(&idx) = self.import_name_to_idx.get(&tf.name) {
                                            self.ir_func_to_idx.insert(*fid, idx);
                                            break 'resolve;
                                        }
                                    }
                                    if let Some(tf) = m2.functions.get(fid) {
                                        if let Some(&idx) = self.import_name_to_idx.get(&tf.name) {
                                            self.ir_func_to_idx.insert(*fid, idx);
                                            break 'resolve;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let (params, results) = self.wasm_sig_for_internal(func_id, func);
            let type_idx = self.intern_type(params, results);
            let func_idx = self.next_func_idx;
            self.next_func_idx += 1;
            self.internals.push(InternalFunc {
                ir_id: *func_id,
                module_idx: mod_idx,
                type_idx,
                func_idx,
            });
            self.ir_func_to_idx.insert(*func_id, func_idx);
            self.func_name_to_idx.insert(func.name.clone(), func_idx);
            // Also register qualified name (ClassName.method)
            if let Some(ref qn) = func.qualified_name {
                self.func_name_to_idx.insert(qn.clone(), func_idx);
            }
            // Store return type for CallDirect type inference
            let ret_ty = ir_type_to_wasm(&func.signature.return_type);
            self.func_return_types.insert(*func_id, ret_ty);
        }

        // Cross-module extern->internal resolution by name.
        for module in modules {
            for (ext_id, ext) in &module.extern_functions {
                if self.ir_func_to_idx.contains_key(ext_id) {
                    continue;
                }
                if let Some(&idx) = self.func_name_to_idx.get(&ext.name) {
                    self.ir_func_to_idx.insert(*ext_id, idx);
                }
            }
        }

        // Ensure ALL functions with code get name mappings.
        for module in modules {
            for (func_id, func) in &module.functions {
                if !func.cfg.blocks.is_empty() {
                    if let Some(&idx) = self.ir_func_to_idx.get(func_id) {
                        self.func_name_to_idx
                            .entry(func.name.clone())
                            .or_insert(idx);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 2 -- string pool
    // ------------------------------------------------------------------

    fn collect_strings(&mut self, modules: &[&IrModule]) {
        for module in modules {
            // Walk the string pool.
            let mut id = 0u32;
            while let Some(s) = module.string_pool.get(id) {
                self.intern_string(s);
                id += 1;
            }
            // Walk function bodies for IrValue::String constants.
            for func in module.functions.values() {
                for block in func.cfg.blocks.values() {
                    for inst in &block.instructions {
                        if let IrInstruction::Const {
                            value: IrValue::String(ref s),
                            ..
                        } = inst
                        {
                            self.intern_string(s);
                        }
                    }
                }
            }
        }
    }

    /// Build fallback map for unresolved CallDirect targets.
    /// Scans all internal functions' instructions for CallDirect with unknown func_ids,
    /// then resolves them by name across all modules' extern_functions and functions.
    fn build_func_id_fallback(&mut self, modules: &[&IrModule]) {
        use crate::ir::IrInstruction;
        let mut unresolved: std::collections::BTreeSet<IrFunctionId> =
            std::collections::BTreeSet::new();

        // Collect all unresolved CallDirect targets
        for module in modules {
            for func in module.functions.values() {
                for block in func.cfg.blocks.values() {
                    for inst in &block.instructions {
                        if let IrInstruction::CallDirect { func_id: fid, .. } = inst {
                            if !self.ir_func_to_idx.contains_key(fid) {
                                unresolved.insert(*fid);
                            }
                        }
                    }
                }
            }
        }

        // Also check for targets that ARE in ir_func_to_idx but might point to wrong functions
        let mut suspicious = 0u32;
        for module in modules {
            for func in module.functions.values() {
                for block in func.cfg.blocks.values() {
                    for inst in &block.instructions {
                        if let IrInstruction::CallDirect { func_id: fid, .. } = inst {
                            if let Some(&idx) = self.ir_func_to_idx.get(fid) {
                                // Check if the target is an import or an internal function
                                let is_import = (idx as usize) < self.imports.len();
                                if !is_import {
                                    // Target is internal — check if the name matches an import
                                    if let Some(ef) = module.extern_functions.get(fid) {
                                        if self.import_name_to_idx.contains_key(&ef.name) {
                                            // This should be calling an import but calls an internal!
                                            let import_idx = self.import_name_to_idx[&ef.name];
                                            eprintln!(
                                                "[wasm] MISROUTE: func_id {:?} ({}) → idx {} (internal) but should be {} (import)",
                                                fid, ef.name, idx, import_idx
                                            );
                                            self.func_id_fallback.insert(*fid, import_idx);
                                            suspicious += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if suspicious > 0 {
            eprintln!("[wasm] fixed {} misrouted CallDirect targets", suspicious);
        }

        if unresolved.is_empty() {
            return;
        }

        // Resolve by name: find the function name for each unresolved ID, then map to import/func index
        for fid in &unresolved {
            for module in modules {
                if let Some(ef) = module.extern_functions.get(fid) {
                    if let Some(&idx) = self.import_name_to_idx.get(&ef.name) {
                        self.func_id_fallback.insert(*fid, idx);
                        break;
                    }
                    if let Some(&idx) = self.func_name_to_idx.get(&ef.name) {
                        self.func_id_fallback.insert(*fid, idx);
                        break;
                    }
                }
                if let Some(ff) = module.functions.get(fid) {
                    if let Some(&idx) = self.import_name_to_idx.get(&ff.name) {
                        self.func_id_fallback.insert(*fid, idx);
                        break;
                    }
                    if let Some(&idx) = self.func_name_to_idx.get(&ff.name) {
                        self.func_id_fallback.insert(*fid, idx);
                        break;
                    }
                }
            }
        }

        let resolved = self.func_id_fallback.len();
        if resolved > 0 || unresolved.len() > resolved {
            eprintln!(
                "[wasm] func_id_fallback: {}/{} resolved",
                resolved,
                unresolved.len()
            );
        }
    }

    /// Get or allocate a table slot for a function (used by MakeClosure).
    fn get_table_slot(&mut self, func_idx: u32) -> u32 {
        if let Some(&slot) = self.table_entries.get(&func_idx) {
            return slot;
        }
        let slot = self.next_table_slot;
        self.next_table_slot += 1;
        self.table_entries.insert(func_idx, slot);
        slot
    }

    /// Add a string to the data section. Returns a pointer to a HaxeString struct.
    ///
    /// Layout in linear memory:
    ///   [utf8 bytes...NUL]           at data_ptr
    ///   [HaxeString { ptr, len, cap }]  at struct_ptr  (returned)
    ///
    /// HaxeString struct (12 bytes on WASM32):
    ///   { ptr: u32, len: u32, cap: u32 }
    fn intern_string(&mut self, s: &str) -> u32 {
        if let Some(&off) = self.string_offsets.get(s) {
            return off;
        }
        let bytes = s.as_bytes();

        // 1. Store raw UTF-8 bytes + NUL terminator
        self.data_offset = (self.data_offset + 3) & !3;
        let data_ptr = self.data_offset;
        let mut data_payload = Vec::with_capacity(bytes.len() + 1);
        data_payload.extend_from_slice(bytes);
        data_payload.push(0); // NUL terminator
        self.string_entries.push(DataString {
            offset: data_ptr,
            bytes: data_payload,
        });
        self.data_offset += bytes.len() as u32 + 1;

        // 2. Store HaxeString struct: { ptr: u32, len: u32, cap: u32 }
        self.data_offset = (self.data_offset + 3) & !3;
        let struct_ptr = self.data_offset;
        let mut struct_payload = Vec::with_capacity(12);
        struct_payload.extend_from_slice(&(data_ptr as u32).to_le_bytes()); // ptr
        struct_payload.extend_from_slice(&(bytes.len() as u32).to_le_bytes()); // len
        struct_payload.extend_from_slice(&(bytes.len() as u32).to_le_bytes()); // cap
        self.string_entries.push(DataString {
            offset: struct_ptr,
            bytes: struct_payload,
        });
        self.data_offset += 12;

        self.string_offsets.insert(s.to_owned(), struct_ptr);
        struct_ptr
    }

    // ------------------------------------------------------------------
    // Phase 3 -- globals
    // ------------------------------------------------------------------

    fn collect_globals(&mut self, modules: &[&IrModule]) {
        let mut next_global_idx = 1u32; // 0 reserved for __stack_pointer
        for module in modules {
            for (gid, global) in &module.globals {
                if self.ir_global_to_idx.contains_key(gid) {
                    continue;
                }
                let vt = ir_type_to_wasm(&global.ty);
                let init = global
                    .initializer
                    .as_ref()
                    .map(ir_value_to_i64)
                    .unwrap_or(0);
                self.ir_global_to_idx.insert(*gid, next_global_idx);
                self.user_globals.push((*gid, vt, init));
                next_global_idx += 1;
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 4 -- encode the WASM module
    // ------------------------------------------------------------------

    fn encode_with_exports(
        &self,
        modules: &[&IrModule],
        entry_function: Option<&str>,
        extra_exports: &[&str],
    ) -> Result<Vec<u8>, String> {
        let mut wasm_module = Module::new();

        // --- Type section ---
        let mut type_section = TypeSection::new();
        for (params, results) in &self.types {
            type_section
                .ty()
                .function(params.iter().copied(), results.iter().copied());
        }
        wasm_module.section(&type_section);

        // --- Import section ---
        // Import section: "rayzor" for runtime functions, actual module name for @:jsImport.
        // @:jsImport functions use their declared JS module name so the linker preserves
        // them as real WASM imports (not stubs). The JS harness provides the host implementations.
        let mut import_section = ImportSection::new();
        for imp in &self.imports {
            let module = if let Some((js_mod, _)) = self.js_import_modules.get(&imp.name) {
                js_mod.as_str()
            } else {
                "rayzor"
            };
            import_section.import(module, &imp.name, EntityType::Function(imp.type_idx));
        }
        wasm_module.section(&import_section);

        // --- Function section ---
        let mut func_section = FunctionSection::new();
        for internal in &self.internals {
            func_section.function(internal.type_idx);
        }
        // Synthetic `_start` wrapper: every module's `__vtable_init__` /
        // `__init__` must run before the entry, mirroring the AOT main
        // wrapper. Without this nothing ever invoked the init hooks on
        // wasm — the class/interface vtable registries stayed empty, so
        // every iface-to-iface cast (haxe_iface_fat_ptr_build) returned
        // null and the first interface method dispatch trapped with
        // "indirect call type mismatch".
        // All vtable inits run before all module inits (same order the AOT
        // main wrapper uses) so __init__ bodies can already go through
        // interface dispatch.
        let mut vtable_init_idxs: Vec<u32> = Vec::new();
        let mut module_init_idxs: Vec<u32> = Vec::new();
        for internal in &self.internals {
            if let Some(func) = modules
                .get(internal.module_idx)
                .and_then(|m| m.functions.get(&internal.ir_id))
            {
                match func.name.as_str() {
                    "__vtable_init__" => vtable_init_idxs.push(internal.func_idx),
                    "__init__" => module_init_idxs.push(internal.func_idx),
                    _ => {}
                }
            }
        }
        let startup_init_idxs: Vec<u32> = vtable_init_idxs
            .into_iter()
            .chain(module_init_idxs)
            .collect();
        // The wrapper sits one past the last allocated index. It is never
        // address-taken, so it needs no table slot and next_func_idx (used
        // for table sizing) stays untouched.
        let void_type_idx = self.type_map.get(&(vec![], vec![])).copied();
        let start_wrapper_idx: Option<u32> = match (void_type_idx, entry_function) {
            (Some(ty), Some(_)) => {
                func_section.function(ty);
                Some(self.next_func_idx)
            }
            _ => None,
        };
        wasm_module.section(&func_section);

        // --- Table section (for call_indirect / closures) ---
        // Always emit a table large enough for all functions — closures store
        // raw func indices and JS needs to call them via table.get().
        let total_funcs = self.next_func_idx;
        {
            let mut table_section = wasm_encoder::TableSection::new();
            table_section.table(wasm_encoder::TableType {
                element_type: wasm_encoder::RefType::FUNCREF,
                minimum: total_funcs as u64,
                maximum: Some(total_funcs as u64),
                table64: false,
                shared: false,
            });
            wasm_module.section(&table_section);
        }

        // --- Memory section ---
        let mut mem_section = MemorySection::new();
        mem_section.memory(MemoryType {
            minimum: INITIAL_MEMORY_PAGES,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        wasm_module.section(&mem_section);

        // --- Global section ---
        let mut global_section = GlobalSection::new();
        // Global 0: __stack_pointer (mutable i32).
        global_section.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: true,
                shared: false,
            },
            &ConstExpr::i32_const(STACK_POINTER_INIT),
        );
        for (_gid, vt, init) in &self.user_globals {
            let expr = match vt {
                ValType::I32 => ConstExpr::i32_const(*init as i32),
                ValType::I64 => ConstExpr::i64_const(*init),
                ValType::F32 => ConstExpr::f32_const(Ieee32::from(*init as f32)),
                ValType::F64 => ConstExpr::f64_const(Ieee64::from(*init as f64)),
                _ => ConstExpr::i32_const(*init as i32),
            };
            global_section.global(
                GlobalType {
                    val_type: *vt,
                    mutable: true,
                    shared: false,
                },
                &expr,
            );
        }
        wasm_module.section(&global_section);

        // --- Export section ---
        let mut entry_idx_for_wrapper: Option<u32> = None;
        let mut export_section = ExportSection::new();
        export_section.export("memory", ExportKind::Memory, 0);
        export_section.export("__indirect_function_table", ExportKind::Table, 0);
        if let Some(entry_name) = entry_function {
            // Search by name in func_name_to_idx
            let idx = self
                .func_name_to_idx
                .get(entry_name)
                .or_else(|| self.func_name_to_idx.get(&format!("@{}", entry_name)))
                .or_else(|| {
                    self.func_name_to_idx
                        .iter()
                        .find(|(k, _)| k.ends_with(entry_name) || k.as_str() == entry_name)
                        .map(|(_, v)| v)
                })
                .copied()
                .or_else(|| {
                    // Search internal functions by scanning the MIR directly
                    for internal in &self.internals {
                        let func = modules
                            .get(internal.module_idx)
                            .and_then(|m| m.functions.get(&internal.ir_id));
                        if let Some(func) = func {
                            if func.name == entry_name || func.name.ends_with(entry_name) {
                                return Some(internal.func_idx);
                            }
                        }
                    }
                    None
                });
            if let Some(idx) = idx {
                if let Some(wrapper) = start_wrapper_idx {
                    entry_idx_for_wrapper = Some(idx);
                    export_section.export("_start", ExportKind::Func, wrapper);
                } else {
                    export_section.export("_start", ExportKind::Func, idx);
                }
            }
        }

        // Export additional functions (for JS interop / @:export)
        for &export_name in extra_exports {
            if let Some(&idx) = self.func_name_to_idx.get(export_name) {
                export_section.export(export_name, ExportKind::Func, idx);
            } else {
                // Try suffix match
                for (name, &idx) in &self.func_name_to_idx {
                    if name.ends_with(export_name) {
                        export_section.export(export_name, ExportKind::Func, idx);
                        break;
                    }
                }
            }
        }

        // Auto-export functions with "export_" prefix (strip prefix)
        // AND functions marked with @:export (use qualified_name as ClassName_method)
        for module in modules {
            for (func_id, func) in &module.functions {
                // Find WASM function index — try func_id first, then qualified_name lookup
                let idx = self.ir_func_to_idx.get(func_id).copied().or_else(|| {
                    func.qualified_name
                        .as_ref()
                        .and_then(|qn| self.func_name_to_idx.get(qn).copied())
                });

                if let Some(idx) = idx {
                    if func.name.starts_with("export_") {
                        let export_name = &func.name["export_".len()..];
                        export_section.export(export_name, ExportKind::Func, idx);
                    } else if func.wasm_export {
                        let export_name = func
                            .qualified_name
                            .as_deref()
                            .unwrap_or(&func.name)
                            .replace('.', "_");
                        export_section.export(&export_name, ExportKind::Func, idx);
                    } else if func.name == "new" {
                        // Constructor: export if class is in exported_classes
                        if let Some(ref qn) = func.qualified_name {
                            if let Some(dot) = qn.rfind('.') {
                                let class = &qn[..dot];
                                if self.exported_classes.contains(class) {
                                    let export_name = qn.replace('.', "_");
                                    export_section.export(&export_name, ExportKind::Func, idx);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Export malloc/free for JS class constructors (needed by bindgen)
        if !self.exported_classes.is_empty() {
            // Try rayzor_malloc first (runtime naming), then malloc
            for name in &["rayzor_malloc", "malloc", "rt_alloc"] {
                if let Some(&idx) = self.func_name_to_idx.get(*name) {
                    export_section.export("malloc", ExportKind::Func, idx);
                    break;
                }
            }
            for name in &["rayzor_free", "free", "_rt_free"] {
                if let Some(&idx) = self.func_name_to_idx.get(*name) {
                    export_section.export("free", ExportKind::Func, idx);
                    break;
                }
            }
        }

        wasm_module.section(&export_section);

        // --- Element section: map ALL function indices into the table ---
        // Closures store raw func indices and call_indirect/JS needs table access.
        // Element (section 9) must come AFTER Export (section 7).
        {
            let mut elem_section = wasm_encoder::ElementSection::new();
            let all_funcs: Vec<u32> = (0..total_funcs).collect();
            elem_section.active(
                Some(0),
                &wasm_encoder::ConstExpr::i32_const(0),
                wasm_encoder::Elements::Functions(std::borrow::Cow::Owned(all_funcs)),
            );
            wasm_module.section(&elem_section);
        }

        // --- Code section ---
        let mut code_section = CodeSection::new();
        for internal in &self.internals {
            let ir_func = modules
                .get(internal.module_idx)
                .and_then(|m| m.functions.get(&internal.ir_id))
                .ok_or_else(|| {
                    format!(
                        "WASM backend: function {:?} not found in modules",
                        internal.ir_id
                    )
                })?;
            // Check if this function should forward to an import.
            // Any function whose name matches a qualified_to_import entry is a stub
            // wrapper that should delegate to the real import.
            // Check if this function should be replaced with a thin forwarder to an import.
            // Only for true stubs (all-Unreachable terminators) AND static-like functions
            // (same param count as the import). Instance method wrappers have complex bodies
            // that extract handles from `this` — those must compile normally.
            let all_unreachable = !ir_func.cfg.blocks.is_empty()
                && ir_func.cfg.blocks.values().all(|b| {
                    matches!(
                        b.terminator,
                        crate::ir::IrTerminator::Unreachable
                            | crate::ir::IrTerminator::NoReturn { .. }
                    )
                });
            let has_import = all_unreachable
                && (self.import_name_to_idx.contains_key(&ir_func.name)
                    || self.qualified_to_import.contains_key(&ir_func.name)
                    || ir_func
                        .qualified_name
                        .as_ref()
                        .map_or(false, |qn| self.qualified_to_import.contains_key(qn)));
            // Also check: if ALL blocks have Unreachable/NoReturn terminator,
            // this is an unresolved extern wrapper. Try to match by method name.
            let has_import = has_import || {
                let all_unreachable = !ir_func.cfg.blocks.is_empty()
                    && ir_func.cfg.blocks.values().all(|b| {
                        matches!(
                            b.terminator,
                            crate::ir::IrTerminator::Unreachable
                                | crate::ir::IrTerminator::NoReturn { .. }
                        )
                    });
                if all_unreachable {
                    let method = ir_func.name.rsplit('.').next().unwrap_or(&ir_func.name);
                    // Search qualified_to_import for any entry ending with this method
                    self.qualified_to_import
                        .keys()
                        .any(|k| k.rsplit('.').next() == Some(method))
                        || self
                            .import_name_to_idx
                            .keys()
                            .any(|k| k.rsplit('_').next() == Some(method))
                } else {
                    false
                }
            };
            if has_import {
                // Try direct name, qualified name, bare method name, then CallDirect scan
                let import_idx = self
                    .import_name_to_idx
                    .get(&ir_func.name)
                    .copied()
                    .or_else(|| self.qualified_to_import.get(&ir_func.name).copied())
                    .or_else(|| {
                        ir_func
                            .qualified_name
                            .as_ref()
                            .and_then(|qn| self.qualified_to_import.get(qn).copied())
                    })
                    // Match by bare method name: "rayzor.concurrent.Mutex.unlock" → find "*.unlock"
                    .or_else(|| {
                        let method = ir_func.name.rsplit('.').next()?;
                        // Match by bare method name across all qualified_to_import entries.
                        // No class prefix filter — wrapper class may differ from registered class
                        // (e.g., Mutex.unlock vs MutexGuard.unlock).
                        self.qualified_to_import
                            .iter()
                            .find(|(k, _)| k.rsplit('.').next() == Some(method))
                            .map(|(_, &idx)| idx)
                    })
                    .or_else(|| {
                        use crate::ir::IrInstruction;
                        for block in ir_func.cfg.blocks.values() {
                            for inst in &block.instructions {
                                if let IrInstruction::CallDirect { func_id: fid, .. } = inst {
                                    if let Some(&idx) = self.ir_func_to_idx.get(fid) {
                                        return Some(idx);
                                    }
                                    if let Some(&idx) = self.func_id_fallback.get(fid) {
                                        return Some(idx);
                                    }
                                }
                            }
                        }
                        None
                    });
                if let Some(import_idx) = import_idx {
                    // Generate a thin forwarder that calls the import with type coercion.
                    let (stub_params, stub_results) =
                        Self::sig_to_wasm_named(&ir_func.name, &ir_func.signature);
                    let (import_params, import_results) =
                        if (import_idx as usize) < self.imports.len() {
                            let imp = &self.imports[import_idx as usize];
                            self.types
                                .get(imp.type_idx as usize)
                                .cloned()
                                .unwrap_or_default()
                        } else {
                            (stub_params.clone(), stub_results.clone())
                        };
                    let mut func = Function::new(std::iter::empty::<(u32, ValType)>());
                    // Pass params with type coercion (f64→i32 conversion for flattened signatures)
                    let n_args = stub_params.len().min(import_params.len());
                    for i in 0..n_args {
                        func.instruction(&Instruction::LocalGet(i as u32));
                        // Coerce type if needed
                        match (stub_params[i], import_params[i]) {
                            (a, b) if a == b => {} // same type, no conversion
                            (ValType::F64, ValType::I32) => {
                                func.instruction(&Instruction::I32TruncF64S);
                            }
                            (ValType::F64, ValType::I64) => {
                                func.instruction(&Instruction::I64TruncF64S);
                            }
                            (ValType::I32, ValType::F64) => {
                                func.instruction(&Instruction::F64ConvertI32S);
                            }
                            (ValType::I64, ValType::F64) => {
                                func.instruction(&Instruction::F64ConvertI64S);
                            }
                            (ValType::F32, ValType::I32) => {
                                func.instruction(&Instruction::I32TruncF32S);
                            }
                            (ValType::I32, ValType::F32) => {
                                func.instruction(&Instruction::F32ConvertI32S);
                            }
                            _ => {} // other conversions: pass through
                        }
                    }
                    // Pad missing import args with zeros
                    for i in n_args..import_params.len() {
                        emit_zero(&mut func, import_params[i]);
                    }
                    func.instruction(&Instruction::Call(import_idx));
                    // Handle result mismatch
                    if import_results.len() > stub_results.len() {
                        for _ in 0..(import_results.len() - stub_results.len()) {
                            func.instruction(&Instruction::Drop);
                        }
                    }
                    for i in import_results.len()..stub_results.len() {
                        emit_zero(&mut func, stub_results[i]);
                    }
                    func.instruction(&Instruction::End);
                    code_section.function(&func);
                    continue;
                }
            }
            let body = FunctionLowerer::lower(self, ir_func)?;
            code_section.function(&body);
        }
        if let Some(_wrapper) = start_wrapper_idx {
            let mut func = Function::new(vec![]);
            for &init_idx in &startup_init_idxs {
                func.instruction(&Instruction::Call(init_idx));
            }
            if let Some(entry_idx) = entry_idx_for_wrapper {
                func.instruction(&Instruction::Call(entry_idx));
            }
            func.instruction(&Instruction::End);
            code_section.function(&func);
        }
        wasm_module.section(&code_section);

        // --- Data section ---
        let mut data_section = DataSection::new();
        for entry in &self.string_entries {
            data_section.active(
                0,
                &ConstExpr::i32_const(entry.offset as i32),
                entry.bytes.iter().copied(),
            );
        }
        wasm_module.section(&data_section);

        // --- Name section (debug aid) ---
        // Function names let trap backtraces resolve to Haxe symbols; the
        // linker remaps and re-emits these into the merged module.
        {
            let mut names = wasm_encoder::NameSection::new();
            let mut fn_names = wasm_encoder::NameMap::new();
            let mut tagged: Vec<(u32, String)> = Vec::new();
            for internal in &self.internals {
                if let Some(func) = modules
                    .get(internal.module_idx)
                    .and_then(|m| m.functions.get(&internal.ir_id))
                {
                    let label = func
                        .qualified_name
                        .clone()
                        .unwrap_or_else(|| func.name.clone());
                    tagged.push((internal.func_idx, label));
                }
            }
            if let Some(wrapper) = start_wrapper_idx {
                tagged.push((wrapper, "__start_wrapper__".to_string()));
            }
            tagged.sort_by_key(|(idx, _)| *idx);
            for (idx, label) in &tagged {
                fn_names.append(*idx, label);
            }
            names.functions(&fn_names);
            wasm_module.section(&names);
        }

        Ok(wasm_module.finish())
    }
}

// ===========================================================================
// Per-function lowering
// ===========================================================================

struct FunctionLowerer<'a> {
    ctx: &'a CompileCtx,
    ir_func: &'a IrFunction,

    /// IrId -> WASM local index.
    reg_to_local: BTreeMap<IrId, u32>,

    /// Locals to declare (count, type) groups. Built during allocation.
    local_types: Vec<ValType>,

    /// Number of WASM parameters (locals 0..param_count-1).
    param_count: u32,
    /// Whether local 0 is an implicit closure env pointer, not a MIR parameter.
    has_implicit_env_param: bool,

    /// Next available local index.
    next_local: u32,

    /// Ordered block IDs (entry first, then sorted by id).
    block_order: Vec<IrBlockId>,
    /// Block ID -> positional index in block_order.
    block_index: BTreeMap<IrBlockId, u32>,
    /// Local holding the shadow-stack pointer captured at function entry.
    /// Restored before every return — without this every Alloc/closure/
    /// struct allocation leaked shadow-stack space permanently (the SP
    /// global only ever descended), eventually marching into the runtime
    /// data segments during model load.
    saved_sp_local: u32,
    /// Reclamation (Phase 1): (block, instruction-index) -> regs whose heap
    /// HaxeString to free immediately AFTER that instruction (its last use).
    /// Computed by `compute_string_free_points`.
    free_points: BTreeMap<(IrBlockId, usize), Vec<IrId>>,
}

impl<'a> FunctionLowerer<'a> {
    fn lower(ctx: &'a CompileCtx, ir_func: &'a IrFunction) -> Result<Function, String> {
        let has_implicit_env_param = ctx.functions_with_env.contains(&ir_func.id);
        let param_offset = if has_implicit_env_param { 1 } else { 0 };
        let param_count = ir_func.signature.parameters.len() as u32 + param_offset;
        let mut low = Self {
            ctx,
            ir_func,
            reg_to_local: BTreeMap::new(),
            local_types: Vec::new(),
            param_count,
            has_implicit_env_param,
            next_local: param_count,
            block_order: Vec::new(),
            block_index: BTreeMap::new(),
            saved_sp_local: 0,
            free_points: BTreeMap::new(),
        };

        // Early return for empty-body functions (extern stubs).
        if ir_func.cfg.blocks.is_empty() {
            return low.build_body();
        }

        // Map parameter registers.
        for (i, param) in ir_func.signature.parameters.iter().enumerate() {
            low.reg_to_local.insert(param.reg, i as u32 + param_offset);
        }

        // Pre-allocate locals for every register used in the function.
        low.allocate_locals();

        // One extra i32 local to save/restore the shadow-stack pointer.
        low.saved_sp_local = low.next_local;
        low.next_local += 1;
        low.local_types.push(ValType::I32);

        // Compute block order.
        low.compute_block_order();

        // Reclamation (Phase 1): compute where to free transient heap strings.
        low.compute_string_free_points();

        // Build the function body.
        low.build_body()
    }

    // ------------------------------------------------------------------
    // WASM memory reclamation — Phase 1: transient heap strings
    //
    // The wasm backend emits no frees, so every per-token string the decode
    // loop produces leaks; the heap climbs until an allocation crosses 2^31
    // and `rt_alloc`'s `ptr as i32` sign-flips into a wild signed write
    // (the long-gen corruption). This pass reclaims the SAFE subset: results
    // of runtime string allocators (heap struct+data) whose entire lifetime
    // is straight-line within one block and which are only ever consumed by
    // non-retaining runtime calls. Anything that escapes (stored, returned,
    // captured, phi'd/loop-carried, aliased, used across blocks, or passed to
    // a user function) is left alone — conservative by construction so a
    // missed case leaks rather than use-after-frees.
    // ------------------------------------------------------------------

    /// Resolve a CallDirect callee's IrFunctionId to its MIR name.
    fn callee_name(&self, func_id: &IrFunctionId) -> Option<&str> {
        for m in self.ctx.all_module_funcs.iter() {
            if let Some(name) = m.get(func_id) {
                return Some(name.as_str());
            }
        }
        None
    }

    /// Runtime string functions that return a FRESH heap HaxeString (the 2-arg
    /// returning forms `rt_alloc(12)` the struct + `alloc_string_data` the buffer).
    /// Their results are reclamation candidates. Matches both the bare stdlib
    /// forward-ref names (`string_concat`) and the qualified runtime names.
    fn is_string_allocator(name: &str) -> bool {
        let bare = name.rsplit(['.', '_']).next().unwrap_or(name);
        matches!(
            name,
            "haxe_string_concat"
                | "string_concat"
                | "haxe_string_from_int"
                | "string_from_int"
                | "haxe_string_from_float"
                | "string_from_float"
                | "haxe_string_from_bool"
                | "string_from_bool"
                | "haxe_string_from_char_code"
                | "string_from_char_code"
                | "haxe_string_substring"
                | "string_substring"
                | "haxe_string_substr"
                | "string_substr"
                | "haxe_string_to_upper_case"
                | "haxe_string_to_lower_case"
        ) || matches!(bare, "concat") && name.contains("string")
    }

    /// Runtime functions that READ a string arg (copy/inspect) but never retain
    /// the pointer, plus print/trace sinks. Passing a transient string to one of
    /// these is a legitimate last use; freeing it afterwards is safe. NB: none of
    /// these return their string argument's pointer (allocators copy; the rest
    /// return ints/void), so the freed buffer can never alias a live result.
    fn is_nonretaining_consumer(name: &str) -> bool {
        Self::is_string_allocator(name)
            || matches!(
                name,
                "haxe_string_length"
                    | "string_length"
                    | "haxe_string_char_at"
                    | "string_char_at"
                    | "haxe_string_char_code_at"
                    | "string_char_code_at"
                    | "haxe_string_index_of"
                    | "string_index_of"
                    | "haxe_string_compare"
                    | "string_compare"
                    | "haxe_string_equals"
                    | "string_equals"
                    | "haxe_sys_println"
                    | "haxe_sys_print"
                    | "sys_println"
                    | "sys_print"
                    | "haxe_trace_string_struct"
            )
    }

    /// Registers an IrTerminator reads (kept local to avoid coupling to Ir internals).
    fn terminator_uses(t: &IrTerminator) -> Vec<IrId> {
        match t {
            IrTerminator::CondBranch { condition, .. } => vec![*condition],
            IrTerminator::Switch { value, .. } => vec![*value],
            IrTerminator::Return { value } => value.map(|v| vec![v]).unwrap_or_default(),
            IrTerminator::NoReturn { call } => vec![*call],
            IrTerminator::Branch { .. } | IrTerminator::Unreachable => Vec::new(),
        }
    }

    fn compute_string_free_points(&mut self) {
        if !self.ctx.wasm_free_enabled || self.ctx.string_free_idx.is_none() {
            return;
        }

        // 1. Candidates: dest of a CallDirect to a string allocator, with its def block.
        let debug = std::env::var_os("RAYZOR_WASM_FREE_DEBUG").is_some();
        let mut seen_stringish: BTreeSet<String> = BTreeSet::new();
        let mut cand_def_block: BTreeMap<IrId, IrBlockId> = BTreeMap::new();
        for (&bid, block) in &self.ir_func.cfg.blocks {
            for inst in &block.instructions {
                if let IrInstruction::CallDirect {
                    dest: Some(d),
                    func_id,
                    ..
                } = inst
                {
                    if let Some(name) = self.callee_name(func_id) {
                        if debug
                            && (name.contains("string")
                                || name.contains("String")
                                || name.contains("concat")
                                || name.contains("substr")
                                || name.contains("from_int")
                                || name.contains("from_float"))
                        {
                            seen_stringish.insert(name.to_string());
                        }
                        if Self::is_string_allocator(name) {
                            cand_def_block.insert(*d, bid);
                        }
                    }
                }
            }
        }
        if debug && !seen_stringish.is_empty() {
            eprintln!(
                "[wasm-free] fn '{}': string-ish callees seen = {:?}; matched candidates = {}",
                self.ir_func.name,
                seen_stringish,
                cand_def_block.len()
            );
        }
        if cand_def_block.is_empty() {
            return;
        }

        let mut disq: BTreeSet<IrId> = BTreeSet::new();
        // candidate -> (block, instruction index) of its last (highest-index) use.
        let mut last_use: BTreeMap<IrId, (IrBlockId, usize)> = BTreeMap::new();

        // 2a. Any phi incoming that is a candidate => merged/loop-carried => disqualify.
        for block in self.ir_func.cfg.blocks.values() {
            for phi in &block.phi_nodes {
                for (_pred, val) in &phi.incoming {
                    if cand_def_block.contains_key(val) {
                        disq.insert(*val);
                    }
                }
            }
        }

        // 2b. Walk instructions: alias/store/capture disqualifiers + use analysis.
        for (&bid, block) in &self.ir_func.cfg.blocks {
            for (idx, inst) in block.instructions.iter().enumerate() {
                // Aliasing / escape into memory or aggregates disqualifies the operand.
                match inst {
                    IrInstruction::Copy { src, .. }
                    | IrInstruction::Move { src, .. }
                    | IrInstruction::Cast { src, .. }
                    | IrInstruction::BitCast { src, .. }
                    | IrInstruction::Clone { src, .. } => {
                        disq.insert(*src);
                    }
                    IrInstruction::Store { value, .. }
                    | IrInstruction::InsertValue { value, .. } => {
                        disq.insert(*value);
                    }
                    IrInstruction::MemCopy { src, .. } => {
                        disq.insert(*src);
                    }
                    IrInstruction::CreateStruct { fields, .. } => {
                        for fv in fields {
                            disq.insert(*fv);
                        }
                    }
                    IrInstruction::MakeClosure {
                        captured_values, ..
                    } => {
                        for cv in captured_values {
                            disq.insert(*cv);
                        }
                    }
                    _ => {}
                }

                // Use analysis: a candidate may only be consumed by a non-retaining
                // runtime call, in its own def block. Record the last such use.
                let nonretaining = match inst {
                    IrInstruction::CallDirect { func_id, .. } => self
                        .callee_name(func_id)
                        .map(Self::is_nonretaining_consumer)
                        .unwrap_or(false),
                    _ => false,
                };
                for u in inst.uses() {
                    let Some(&def_bid) = cand_def_block.get(&u) else {
                        continue;
                    };
                    if bid != def_bid || !nonretaining {
                        disq.insert(u);
                        continue;
                    }
                    let e = last_use.entry(u).or_insert((bid, idx));
                    if idx > e.1 {
                        *e = (bid, idx);
                    }
                }
            }
            // Terminator uses => cross-block / return liveness => disqualify.
            for u in Self::terminator_uses(&block.terminator) {
                if cand_def_block.contains_key(&u) {
                    disq.insert(u);
                }
            }
        }

        // 3. Survivors with a recorded last use get a free point right after it.
        let mut points: BTreeMap<(IrBlockId, usize), Vec<IrId>> = BTreeMap::new();
        for cand in cand_def_block.keys() {
            if disq.contains(cand) {
                continue;
            }
            if let Some(&(bid, idx)) = last_use.get(cand) {
                points.entry((bid, idx)).or_default().push(*cand);
            }
        }
        if debug {
            let freed: usize = points.values().map(|v| v.len()).sum();
            eprintln!(
                "[wasm-free] fn '{}': {} candidates, {} freed at last-use",
                self.ir_func.name,
                cand_def_block.len(),
                freed
            );
        }
        self.free_points = points;
    }

    /// Emit `haxe_string_free(reg)` — reclaims the heap data buffer of a HaxeString
    /// whose lifetime has ended. No-op if the free import isn't present.
    fn emit_string_free(&self, f: &mut Function, reg: IrId) {
        if let Some(idx) = self.ctx.string_free_idx {
            self.get_reg(f, reg);
            f.instruction(&Instruction::Call(idx));
        }
    }

    // ------------------------------------------------------------------
    // Local allocation
    // ------------------------------------------------------------------

    fn allocate_locals(&mut self) {
        // Step 1: Allocate all locals, using register_types for initial type when available.
        // This ensures F64 registers get F64 locals from the start, preventing
        // type mismatches in Copy/phi resolution that bypass producer inference.
        let mut seen = BTreeSet::new();
        for p in &self.ir_func.signature.parameters {
            seen.insert(p.reg);
        }
        for block in self.ir_func.cfg.blocks.values() {
            for phi in &block.phi_nodes {
                if seen.insert(phi.dest) {
                    let ty = self
                        .ir_func
                        .register_types
                        .get(&phi.dest)
                        .map(|t| ir_type_to_wasm(t))
                        .unwrap_or(ValType::I32);
                    self.alloc_local(phi.dest, ty);
                }
            }
            for inst in &block.instructions {
                if let Some(dest) = inst.dest() {
                    if seen.insert(dest) {
                        // Use the actual value type for Const/BitCast instructions — the MIR
                        // register_types may say I32/PtrVoid even for F64 constants.
                        let ty = match inst {
                            IrInstruction::Const { value, .. } => match value {
                                IrValue::F32(_) => ValType::F32,
                                IrValue::F64(_) => ValType::F64,
                                _ => ValType::I32,
                            },
                            // BitCast to I64/U64 from F64: need I64 local to hold 64 bits
                            IrInstruction::BitCast { ty, .. } => match ty {
                                IrType::I64 | IrType::U64 => ValType::I64,
                                _ => ir_type_to_wasm(ty),
                            },
                            _ => self
                                .ir_func
                                .register_types
                                .get(&dest)
                                .map(|t| ir_type_to_wasm(t))
                                .unwrap_or(ValType::I32),
                        };
                        self.alloc_local(dest, ty);
                    }
                }
            }
        }

        // Step 2: Producer-based type inference.
        // Determine each register's type from what PRODUCES the value.
        // Iterate to fixpoint (needed for BinOp depending on other BinOp).
        // Only UPGRADE from I32 to F32/F64 — never downgrade.
        loop {
            let mut changed = false;
            for block in self.ir_func.cfg.blocks.values() {
                // Phi: type from phi.ty annotation
                for phi in &block.phi_nodes {
                    let ty = ir_type_to_wasm(&phi.ty);
                    if ty != ValType::I32 && self.local_type_of(phi.dest) != Some(ty) {
                        self.set_local_type(phi.dest, ty);
                        changed = true;
                    }
                }

                for inst in &block.instructions {
                    let produced_ty = match inst {
                        // Instructions with explicit type annotations
                        IrInstruction::Load { ty, .. } => Some(ir_type_to_wasm(ty)),
                        IrInstruction::Undef { ty, .. } => Some(ir_type_to_wasm(ty)),
                        IrInstruction::Cast { to_ty, .. } => Some(ir_type_to_wasm(to_ty)),

                        // Constants carry their type
                        IrInstruction::Const { value, .. } => Some(match value {
                            IrValue::F32(_) => ValType::F32,
                            IrValue::F64(_) => ValType::F64,
                            _ => ValType::I32,
                        }),

                        // Comparisons always produce i32 (boolean)
                        IrInstruction::Cmp { .. } => Some(ValType::I32),

                        // CallDirect: use callee return type, fallback to dest register type
                        IrInstruction::CallDirect { dest, func_id, .. } => self
                            .ctx
                            .func_return_types
                            .get(func_id)
                            .copied()
                            .or_else(|| {
                                dest.and_then(|d| self.ir_func.register_types.get(&d))
                                    .map(|t| ir_type_to_wasm(t))
                            }),

                        // BinOp: choose result type from operand types. Prefer the
                        // highest precision F64 operand; otherwise F32 if any operand
                        // is F32 (or it's a float op), otherwise I32.
                        IrInstruction::BinOp {
                            op, left, right, ..
                        } => {
                            let is_fop = matches!(
                                op,
                                BinaryOp::FAdd
                                    | BinaryOp::FSub
                                    | BinaryOp::FMul
                                    | BinaryOp::FDiv
                                    | BinaryOp::FRem
                            );
                            let lt = self.local_type_of(*left).unwrap_or(ValType::I32);
                            let rt = self.local_type_of(*right).unwrap_or(ValType::I32);
                            if lt == ValType::F64 || rt == ValType::F64 {
                                Some(ValType::F64)
                            } else if is_fop || lt == ValType::F32 || rt == ValType::F32 {
                                Some(ValType::F32)
                            } else {
                                None // keep I32
                            }
                        }

                        // UnOp: inherit operand type. Prefer operand precision over
                        // unconditional F64 promotion.
                        IrInstruction::UnOp { op, operand, .. } => {
                            let is_fop = matches!(op, UnaryOp::FNeg);
                            let ot = self.local_type_of(*operand).unwrap_or(ValType::I32);
                            if ot == ValType::F64 {
                                Some(ValType::F64)
                            } else if is_fop || ot == ValType::F32 {
                                Some(ValType::F32)
                            } else {
                                None
                            }
                        }

                        // CallIndirect: use dest register type
                        IrInstruction::CallIndirect { dest, .. } => dest
                            .and_then(|d| self.ir_func.register_types.get(&d))
                            .map(|t| ir_type_to_wasm(t)),

                        // GEP, PtrAdd, Alloc: always i32 (pointer)
                        IrInstruction::GetElementPtr { .. }
                        | IrInstruction::PtrAdd { .. }
                        | IrInstruction::Alloc { .. }
                        | IrInstruction::FunctionRef { .. }
                        | IrInstruction::MakeClosure { .. }
                        | IrInstruction::ClosureFunc { .. }
                        | IrInstruction::ClosureEnv { .. }
                        | IrInstruction::CreateStruct { .. }
                        | IrInstruction::CreateUnion { .. }
                        | IrInstruction::ExtractDiscriminant { .. } => Some(ValType::I32),

                        // Copy/Move: inherit SOURCE type (not register_types)
                        IrInstruction::Copy { src, .. }
                        | IrInstruction::Move { src, .. }
                        | IrInstruction::Clone { src, .. }
                        | IrInstruction::BorrowImmutable { src, .. }
                        | IrInstruction::BorrowMutable { src, .. } => self.local_type_of(*src),

                        // Select: float if either branch is float
                        IrInstruction::Select {
                            true_val,
                            false_val,
                            ..
                        } => {
                            let tv = self.local_type_of(*true_val).unwrap_or(ValType::I32);
                            let fv = self.local_type_of(*false_val).unwrap_or(ValType::I32);
                            if tv == ValType::F64 || fv == ValType::F64 {
                                Some(ValType::F64)
                            } else if tv == ValType::F32 || fv == ValType::F32 {
                                Some(ValType::F32)
                            } else {
                                None
                            }
                        }

                        _ => None,
                    };

                    if let Some(ty) = produced_ty {
                        if let Some(dest) = inst.dest() {
                            if ty != ValType::I32 && self.local_type_of(dest) != Some(ty) {
                                self.set_local_type(dest, ty);
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Step 3: Final sweep — use register_types as authoritative fallback.
        // Catches cases where MIR type info is more specific than producer inference
        // (e.g., extern function wrappers returning F64 via PtrVoid MIR type).
        // Also scan Const instructions directly to catch F64 consts in I32-typed registers.
        for block in self.ir_func.cfg.blocks.values() {
            for inst in &block.instructions {
                if let IrInstruction::Const { dest, value } = inst {
                    let const_ty = match value {
                        IrValue::F32(_) => ValType::F32,
                        IrValue::F64(_) => ValType::F64,
                        _ => ValType::I32,
                    };
                    if const_ty != ValType::I32 {
                        let current = self.local_type_of(*dest);
                        if current == Some(ValType::I32) || current.is_none() {
                            if current.is_some() {
                                self.set_local_type(*dest, const_ty);
                            }
                        }
                    }
                }
            }
            // Also upgrade phi dests if any incoming value is float
            for phi in &block.phi_nodes {
                for (_pred, val) in &phi.incoming {
                    if let Some(vt) = self.local_type_of(*val) {
                        if vt != ValType::I32 {
                            if let Some(dt) = self.local_type_of(phi.dest) {
                                if dt == ValType::I32 {
                                    self.set_local_type(phi.dest, vt);
                                }
                            }
                        }
                    }
                }
            }
        }
        for (&reg, mir_ty) in &self.ir_func.register_types {
            let wt = ir_type_to_wasm(mir_ty);
            if wt != ValType::I32 {
                if let Some(current) = self.local_type_of(reg) {
                    if current == ValType::I32 {
                        self.set_local_type(reg, wt);
                    }
                }
            }
        }
    }

    /// Get the WASM type of an IrId's allocated local.
    fn local_type_of(&self, id: IrId) -> Option<ValType> {
        let &local_idx = self.reg_to_local.get(&id)?;
        if local_idx < self.param_count {
            if self.has_implicit_env_param && local_idx == 0 {
                return Some(ValType::I32);
            }
            let param_idx = local_idx as usize;
            let param_idx = if self.has_implicit_env_param {
                param_idx.saturating_sub(1)
            } else {
                param_idx
            };
            self.ir_func
                .signature
                .parameters
                .get(param_idx)
                .map(|p| ir_type_to_wasm(&p.ty))
        } else {
            let type_idx = (local_idx - self.param_count) as usize;
            self.local_types.get(type_idx).copied()
        }
    }

    /// Update an allocated local's type.
    fn set_local_type(&mut self, id: IrId, ty: ValType) {
        if let Some(&local_idx) = self.reg_to_local.get(&id) {
            if local_idx >= self.param_count {
                let type_idx = (local_idx - self.param_count) as usize;
                if type_idx < self.local_types.len() {
                    self.local_types[type_idx] = ty;
                }
            }
        }
    }

    fn alloc_local(&mut self, id: IrId, vt: ValType) -> u32 {
        let idx = self.next_local;
        self.next_local += 1;
        self.local_types.push(vt);
        self.reg_to_local.insert(id, idx);
        idx
    }

    fn alloc_scratch(&mut self, vt: ValType) -> u32 {
        let idx = self.next_local;
        self.next_local += 1;
        self.local_types.push(vt);
        idx
    }

    fn reg_wasm_type(&self, id: IrId) -> ValType {
        // Use the producer-inferred type from allocate_locals.
        // This is the authoritative source — it reflects what the
        // instruction that PRODUCES this value actually generates.
        self.local_type_of(id).unwrap_or(ValType::I32)
    }

    // ------------------------------------------------------------------
    // Block ordering
    // ------------------------------------------------------------------

    fn compute_block_order(&mut self) {
        let entry = self.ir_func.cfg.entry_block;
        self.block_order.push(entry);
        for &bid in self.ir_func.cfg.blocks.keys() {
            if bid != entry {
                self.block_order.push(bid);
            }
        }
        for (i, &bid) in self.block_order.iter().enumerate() {
            self.block_index.insert(bid, i as u32);
        }
    }

    // ------------------------------------------------------------------
    // Body construction
    // ------------------------------------------------------------------

    fn build_body(&mut self) -> Result<Function, String> {
        let n = self.block_order.len();

        if n == 0 {
            // Empty stub — return default value if non-void, otherwise just end.
            let mut f = Function::new(self.local_types.iter().map(|t| (1, *t)));
            if !matches!(self.ir_func.signature.return_type, IrType::Void) {
                emit_zero(&mut f, ir_type_to_wasm(&self.ir_func.signature.return_type));
            }
            f.instruction(&Instruction::End);
            return Ok(f);
        }

        if n == 1 {
            return self.build_single_block();
        }

        self.build_multi_block()
    }

    // ------------------------------------------------------------------
    // Single-block function (no CFG needed)
    // ------------------------------------------------------------------

    fn emit_sp_save(&self, f: &mut Function) {
        f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
        f.instruction(&Instruction::LocalSet(self.saved_sp_local));
    }

    fn emit_sp_restore(&self, f: &mut Function) {
        f.instruction(&Instruction::LocalGet(self.saved_sp_local));
        f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));
    }

    fn build_single_block(&mut self) -> Result<Function, String> {
        let bid = self.block_order[0];
        let block = self.ir_func.cfg.blocks.get(&bid).unwrap();

        let mut f = Function::new(self.local_types.iter().map(|t| (1, *t)));
        self.emit_sp_save(&mut f);

        self.emit_block_instructions(&mut f, bid, &block.instructions);

        self.emit_terminator_simple(&mut f, &block.terminator);
        f.instruction(&Instruction::End);
        Ok(f)
    }

    // ------------------------------------------------------------------
    // Multi-block function: br_table dispatch loop
    //
    //   (local $blk i32)
    //   (block $exit
    //     (loop $dispatch
    //       (block $bN-1
    //         ...
    //           (block $b0
    //             (local.get $blk)
    //             (br_table $b0 $b1 ... $bN-1 $exit)
    //           ) ;; end $b0   -- block 0 code here
    //         ) ;; end $b1     -- block 1 code here
    //       ) ;; end $bN-1     -- block N-1 code here
    //     ) ;; end loop
    //   ) ;; end $exit
    //
    // Within each block's code: set $blk, then br to the loop label.
    // ------------------------------------------------------------------

    fn build_multi_block(&mut self) -> Result<Function, String> {
        let n = self.block_order.len();

        // Allocate the dispatch variable.
        let blk_local = self.alloc_scratch(ValType::I32);

        let mut f = Function::new(self.local_types.iter().map(|t| (1, *t)));
        self.emit_sp_save(&mut f);

        // Initialize $blk to 0 (entry block).
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::LocalSet(blk_local));

        // (block $exit
        f.instruction(&Instruction::Block(BlockType::Empty));
        // (loop $dispatch
        f.instruction(&Instruction::Loop(BlockType::Empty));

        // Nest N blocks.
        for _ in 0..n {
            f.instruction(&Instruction::Block(BlockType::Empty));
        }

        // br_table dispatch.
        f.instruction(&Instruction::LocalGet(blk_local));
        let targets: Vec<u32> = (0..n as u32).collect();
        let default = n as u32 + 1; // -> $exit
        f.instruction(&Instruction::BrTable(Cow::Owned(targets), default));

        // Emit each block: close its Block, emit code, br back to loop.
        for i in 0..n {
            // Close $b_i
            f.instruction(&Instruction::End);

            let bid = self.block_order[i];
            let block = self.ir_func.cfg.blocks.get(&bid).unwrap();

            // Emit instructions (with vector FMA peephole).
            self.emit_block_instructions(&mut f, bid, &block.instructions);

            // Emit terminator (sets $blk, possibly returns).
            let returned = self.emit_terminator_dispatch(&mut f, block, blk_local);

            if !returned {
                // Branch back to loop $dispatch.
                // Current nesting after $b_i end: remaining blocks + loop.
                // Depth to loop = (n - 1 - i).
                let depth_to_loop = (n - 1 - i) as u32;
                f.instruction(&Instruction::Br(depth_to_loop));
            }
        }

        // Close loop.
        f.instruction(&Instruction::End);
        // Close $exit.
        f.instruction(&Instruction::End);

        // If function returns a value, push a default (unreachable path safety).
        // All real returns use `return` inside the dispatch loop.
        let ret_ty = ir_type_to_wasm(&self.ir_func.signature.return_type);
        if !matches!(self.ir_func.signature.return_type, IrType::Void) {
            emit_zero(&mut f, ret_ty);
        }

        // Function end.
        f.instruction(&Instruction::End);

        Ok(f)
    }

    // ------------------------------------------------------------------
    // Terminator: simple (single block)
    // ------------------------------------------------------------------

    fn emit_terminator_simple(&self, f: &mut Function, term: &IrTerminator) {
        match term {
            IrTerminator::Return { value } => {
                if let Some(v) = value {
                    self.get_reg(f, *v);
                    if matches!(self.ir_func.signature.return_type, IrType::Void) {
                        f.instruction(&Instruction::Drop);
                    } else {
                        let actual_ty = self
                            .local_type_of(*v)
                            .unwrap_or_else(|| self.reg_wasm_type(*v));
                        let return_ty = ir_type_to_wasm(&self.ir_func.signature.return_type);
                        emit_bitcast(f, actual_ty, return_ty);
                    }
                } else if !matches!(self.ir_func.signature.return_type, IrType::Void) {
                    emit_zero(f, ir_type_to_wasm(&self.ir_func.signature.return_type));
                }
                self.emit_sp_restore(f);
                f.instruction(&Instruction::Return);
            }
            IrTerminator::Unreachable | IrTerminator::NoReturn { .. } => {
                f.instruction(&Instruction::Unreachable);
            }
            _ => {
                f.instruction(&Instruction::Unreachable);
            }
        }
    }

    // ------------------------------------------------------------------
    // Terminator: dispatch loop (multi block). Returns true if it emitted
    // a `return` or `unreachable` (no need for the caller to emit `br`).
    // ------------------------------------------------------------------

    fn emit_terminator_dispatch(
        &self,
        f: &mut Function,
        block: &IrBasicBlock,
        blk_local: u32,
    ) -> bool {
        let current_bid = block.id;

        match &block.terminator {
            IrTerminator::Return { value } => {
                if let Some(v) = value {
                    self.get_reg(f, *v);
                    if matches!(self.ir_func.signature.return_type, IrType::Void) {
                        f.instruction(&Instruction::Drop);
                    } else {
                        let actual_ty = self
                            .local_type_of(*v)
                            .unwrap_or_else(|| self.reg_wasm_type(*v));
                        let return_ty = ir_type_to_wasm(&self.ir_func.signature.return_type);
                        emit_bitcast(f, actual_ty, return_ty);
                    }
                } else if !matches!(self.ir_func.signature.return_type, IrType::Void) {
                    emit_zero(f, ir_type_to_wasm(&self.ir_func.signature.return_type));
                }
                self.emit_sp_restore(f);
                f.instruction(&Instruction::Return);
                true
            }

            IrTerminator::Branch { target } => {
                self.emit_phi_set(f, current_bid, *target);
                let idx = self.block_index.get(target).copied().unwrap_or(0);
                f.instruction(&Instruction::I32Const(idx as i32));
                f.instruction(&Instruction::LocalSet(blk_local));
                false // caller emits br
            }

            IrTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let true_idx = self.block_index.get(true_target).copied().unwrap_or(0);
                let false_idx = self.block_index.get(false_target).copied().unwrap_or(0);

                self.get_reg(f, *condition);
                f.instruction(&Instruction::If(BlockType::Empty));
                {
                    self.emit_phi_set(f, current_bid, *true_target);
                    f.instruction(&Instruction::I32Const(true_idx as i32));
                    f.instruction(&Instruction::LocalSet(blk_local));
                }
                f.instruction(&Instruction::Else);
                {
                    self.emit_phi_set(f, current_bid, *false_target);
                    f.instruction(&Instruction::I32Const(false_idx as i32));
                    f.instruction(&Instruction::LocalSet(blk_local));
                }
                f.instruction(&Instruction::End);
                false
            }

            IrTerminator::Switch {
                value,
                cases,
                default,
            } => {
                // Chain of if/else -- correct for Phase 1.
                let default_idx = self.block_index.get(default).copied().unwrap_or(0);

                // Set default first.
                self.emit_phi_set(f, current_bid, *default);
                f.instruction(&Instruction::I32Const(default_idx as i32));
                f.instruction(&Instruction::LocalSet(blk_local));

                for (case_val, case_target) in cases {
                    self.get_reg(f, *value);
                    f.instruction(&Instruction::I32Const(*case_val as i32));
                    f.instruction(&Instruction::I32Eq);
                    f.instruction(&Instruction::If(BlockType::Empty));
                    {
                        self.emit_phi_set(f, current_bid, *case_target);
                        let idx = self.block_index.get(case_target).copied().unwrap_or(0);
                        f.instruction(&Instruction::I32Const(idx as i32));
                        f.instruction(&Instruction::LocalSet(blk_local));
                    }
                    f.instruction(&Instruction::End);
                }
                false
            }

            IrTerminator::Unreachable | IrTerminator::NoReturn { .. } => {
                f.instruction(&Instruction::Unreachable);
                true
            }
        }
    }

    // ------------------------------------------------------------------
    // Phi node resolution: set target block's phi locals from current block
    // ------------------------------------------------------------------

    fn emit_phi_set(&self, f: &mut Function, from_bid: IrBlockId, to_bid: IrBlockId) {
        if let Some(target_block) = self.ir_func.cfg.blocks.get(&to_bid) {
            for phi in &target_block.phi_nodes {
                for (pred_bid, val) in &phi.incoming {
                    if *pred_bid == from_bid {
                        self.get_reg(f, *val);
                        let val_ty = self.reg_wasm_type(*val);
                        let dest_ty = self.reg_wasm_type(phi.dest);
                        self.emit_type_coerce(f, val_ty, dest_ty);
                        self.set_reg(f, phi.dest);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Block-level emission with FMA peephole
    //
    // Walks a basic block's instructions and emits each, with one
    // optimisation: vector `fmul; fadd` (and `c - fmul(a,b)`) pairs
    // fuse into `F32x4RelaxedMadd` / `F32x4RelaxedNmadd` from the
    // relaxed-SIMD proposal. Same intent as the Cranelift FMA peephole
    // at instruction_lowering.rs + cranelift_backend.rs (commits
    // 8394d20 + the pre-existing scalar fusion).
    //
    // The peephole maintains a per-block map `vfmul_map: dest →
    // (left, right)` populated by VectorBinOp::Mul/FMul on f32x4 and
    // queried by VectorBinOp::Add/FAdd/Sub/FSub on the same type.
    // When an add/sub's operand was produced by an fmul in this block
    // we emit the fused instruction instead of the separate ops.
    //
    // Caveats:
    // - Map is per-block (cleared at block boundary). Cross-block
    //   fusion would have the same correctness concerns as Cranelift
    //   (SROA + CopyProp can expose fmul results across blocks that
    //   originally went through memory). Same-block keeps semantics
    //   simple.
    // - The original fmul's WASM bytecode IS still emitted (mul +
    //   local.set). wasm-opt's DCE pass should eliminate it when the
    //   local is dead. Single-use analysis at this layer would be a
    //   nice-to-have but adds complexity for marginal binary size.
    // - Relaxed-SIMD is gated on the runtime supporting the proposal.
    //   wasmtime 12+, V8/SpiderMonkey current. If a target environment
    //   doesn't support it, the bytecode won't load — same situation
    //   as the rest of the SIMD lowering, which assumes SIMD128 is
    //   available.
    fn emit_block_instructions(&self, f: &mut Function, bid: IrBlockId, insts: &[IrInstruction]) {
        let mut vfmul_map: std::collections::BTreeMap<IrId, (IrId, IrId)> =
            std::collections::BTreeMap::new();
        for (idx, inst) in insts.iter().enumerate() {
            if let IrInstruction::VectorBinOp {
                dest,
                op,
                left,
                right,
                vec_ty,
            } = inst
            {
                if is_f32x4_vec(vec_ty) {
                    match op {
                        BinaryOp::Mul | BinaryOp::FMul => {
                            // Record the fmul for a potential fuse on the
                            // next f32x4 fadd/fsub that consumes `dest`.
                            // Then fall through to emit the normal mul —
                            // wasm-opt DCEs it if the local is unread.
                            vfmul_map.insert(*dest, (*left, *right));
                        }
                        BinaryOp::Add | BinaryOp::FAdd => {
                            // Fuse: fma(a, b, c) where c is the
                            // non-fmul operand.
                            if let Some(&(a, b)) = vfmul_map.get(left) {
                                self.get_reg(f, a);
                                self.get_reg(f, b);
                                self.get_reg(f, *right);
                                f.instruction(&Instruction::F32x4RelaxedMadd);
                                self.set_reg(f, *dest);
                                continue;
                            }
                            if let Some(&(a, b)) = vfmul_map.get(right) {
                                self.get_reg(f, a);
                                self.get_reg(f, b);
                                self.get_reg(f, *left);
                                f.instruction(&Instruction::F32x4RelaxedMadd);
                                self.set_reg(f, *dest);
                                continue;
                            }
                        }
                        BinaryOp::Sub | BinaryOp::FSub => {
                            // `c - fmul(a, b)` → `-(a*b) + c` →
                            // RelaxedNmadd(a, b, c). Only this orientation
                            // has a single-instruction form; `fmul - c`
                            // would need fma(a, b, -c) which is two ops.
                            if let Some(&(a, b)) = vfmul_map.get(right) {
                                self.get_reg(f, a);
                                self.get_reg(f, b);
                                self.get_reg(f, *left);
                                f.instruction(&Instruction::F32x4RelaxedNmadd);
                                self.set_reg(f, *dest);
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
            }
            self.emit_instruction(f, inst);
            // Reclamation (Phase 1): free transient heap strings whose last use
            // was this instruction. The fused-vector `continue` paths above skip
            // this, but a vector op is never a string-allocator result, so no
            // free point is ever attached to a fused index.
            if let Some(regs) = self.free_points.get(&(bid, idx)) {
                for &reg in regs {
                    self.emit_string_free(f, reg);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Instruction lowering
    // ------------------------------------------------------------------

    fn emit_instruction(&self, f: &mut Function, inst: &IrInstruction) {
        match inst {
            // === Const ===
            IrInstruction::Const { dest, value } => {
                self.emit_const(f, value);
                self.set_reg(f, *dest);
            }

            // === Copy / Move / Borrow / Clone ===
            IrInstruction::Copy { dest, src }
            | IrInstruction::Move { dest, src }
            | IrInstruction::BorrowImmutable { dest, src, .. }
            | IrInstruction::BorrowMutable { dest, src, .. }
            | IrInstruction::Clone { dest, src } => {
                let src_ty = self.reg_wasm_type(*src);
                let dest_ty = self.reg_wasm_type(*dest);
                self.get_reg(f, *src);
                self.emit_type_coerce(f, src_ty, dest_ty);
                self.set_reg(f, *dest);
            }

            // === BinOp ===
            IrInstruction::BinOp {
                dest,
                op,
                left,
                right,
            } => {
                let left_ty = self.reg_wasm_type(*left);
                let right_ty = self.reg_wasm_type(*right);
                let dest_ty = self.reg_wasm_type(*dest);

                // Determine the operation type. For floating-point ops, prefer
                // the highest-precision operand type. Both operands F32 → F32 op;
                // any F64 operand → F64 op (with promotion). Otherwise integer.
                let is_fop = matches!(
                    op,
                    BinaryOp::FAdd
                        | BinaryOp::FSub
                        | BinaryOp::FMul
                        | BinaryOp::FDiv
                        | BinaryOp::FRem
                );
                let op_ty = if dest_ty == ValType::F64
                    || left_ty == ValType::F64
                    || right_ty == ValType::F64
                {
                    ValType::F64
                } else if is_fop
                    || dest_ty == ValType::F32
                    || left_ty == ValType::F32
                    || right_ty == ValType::F32
                {
                    ValType::F32
                } else {
                    ValType::I32
                };

                // Load with coercion
                self.get_reg(f, *left);
                self.emit_type_coerce(f, left_ty, op_ty);
                self.get_reg(f, *right);
                self.emit_type_coerce(f, right_ty, op_ty);

                self.emit_binop(f, *op, op_ty);

                // Coerce result to dest type
                self.emit_type_coerce(f, op_ty, dest_ty);
                self.set_reg(f, *dest);
            }

            // === UnOp ===
            IrInstruction::UnOp { dest, op, operand } => {
                let dest_ty = self.reg_wasm_type(*dest);
                let operand_ty = self.reg_wasm_type(*operand);
                // Promote: if operand is float, use float op even if dest was inferred as i32
                let op_ty = if operand_ty == ValType::F64 || dest_ty == ValType::F64 {
                    ValType::F64
                } else if operand_ty == ValType::F32 || dest_ty == ValType::F32 {
                    ValType::F32
                } else {
                    dest_ty
                };
                self.emit_unop(f, *op, *operand, op_ty);
                self.emit_type_coerce(f, op_ty, dest_ty);
                self.set_reg(f, *dest);
            }

            // === Cmp ===
            IrInstruction::Cmp {
                dest,
                op,
                left,
                right,
            } => {
                let left_ty = self.reg_wasm_type(*left);
                let right_ty = self.reg_wasm_type(*right);
                let is_fcmp = matches!(
                    op,
                    CompareOp::FEq
                        | CompareOp::FNe
                        | CompareOp::FLt
                        | CompareOp::FLe
                        | CompareOp::FGt
                        | CompareOp::FGe
                );
                let cmp_ty = if is_fcmp || left_ty == ValType::F64 || right_ty == ValType::F64 {
                    ValType::F64
                } else if left_ty == ValType::F32 || right_ty == ValType::F32 {
                    ValType::F32
                } else {
                    left_ty
                };

                self.get_reg(f, *left);
                self.emit_type_coerce(f, left_ty, cmp_ty);
                self.get_reg(f, *right);
                self.emit_type_coerce(f, right_ty, cmp_ty);

                self.emit_cmp(f, *op, cmp_ty);
                self.set_reg(f, *dest); // Cmp always produces i32
            }

            // === Cast ===
            IrInstruction::Cast {
                dest,
                src,
                from_ty,
                to_ty,
            } => {
                self.get_reg(f, *src);
                // Use the actual local type for cast source, not MIR's from_ty
                // (the inference pass may have changed the local's type)
                let actual_src_ty = self.reg_wasm_type(*src);
                let target_ty = ir_type_to_wasm(to_ty);
                self.emit_type_coerce(f, actual_src_ty, target_ty);
                self.set_reg(f, *dest);
            }

            // === BitCast ===
            IrInstruction::BitCast { dest, src, ty } => {
                self.get_reg(f, *src);
                let from_vt = self.reg_wasm_type(*src);
                let to_vt = self.reg_wasm_type(*dest);
                // On WASM32, BitCast between F64 and I32 is lossy but must be valid.
                // If the dest is I32 but src is F64, use the cross-width bitcast.
                // If the MIR target type is I64/U64, use reinterpret to keep full bits
                // (only works if the local was allocated as I64).
                emit_bitcast(f, from_vt, to_vt);
                self.set_reg(f, *dest);
            }

            // === Load ===
            IrInstruction::Load { dest, ptr, ty } => {
                self.get_reg(f, *ptr);
                let ma = MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                };
                match ir_type_to_wasm(ty) {
                    ValType::I32 => match ty {
                        IrType::I8 => {
                            f.instruction(&Instruction::I32Load8S(ma));
                        }
                        IrType::U8 | IrType::Bool => {
                            f.instruction(&Instruction::I32Load8U(ma));
                        }
                        IrType::I16 => {
                            f.instruction(&Instruction::I32Load16S(ma));
                        }
                        IrType::U16 => {
                            f.instruction(&Instruction::I32Load16U(ma));
                        }
                        _ => {
                            f.instruction(&Instruction::I32Load(ma));
                        }
                    },
                    ValType::I64 => {
                        f.instruction(&Instruction::I64Load(ma));
                    }
                    ValType::F32 => {
                        f.instruction(&Instruction::F32Load(ma));
                    }
                    ValType::F64 => {
                        f.instruction(&Instruction::F64Load(ma));
                    }
                    _ => {
                        f.instruction(&Instruction::I32Load(ma));
                    }
                };
                self.set_reg(f, *dest);
            }

            // === Store ===
            IrInstruction::Store {
                ptr,
                value,
                store_ty,
            } => {
                self.get_reg(f, *ptr);
                self.get_reg(f, *value);
                let ma = MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                };
                let actual_ty = store_ty.as_ref().unwrap_or(
                    self.ir_func
                        .register_types
                        .get(value)
                        .unwrap_or(&IrType::I32),
                );
                let value_ty = self
                    .ir_func
                    .register_types
                    .get(value)
                    .unwrap_or(&IrType::I32);
                let vt = ir_type_to_wasm(actual_ty);
                let value_vt = self
                    .local_type_of(*value)
                    .unwrap_or_else(|| ir_type_to_wasm(value_ty));
                emit_bitcast(f, value_vt, vt);
                match vt {
                    ValType::I32 => match actual_ty {
                        IrType::I8 | IrType::U8 | IrType::Bool => {
                            f.instruction(&Instruction::I32Store8(ma));
                        }
                        IrType::I16 | IrType::U16 => {
                            f.instruction(&Instruction::I32Store16(ma));
                        }
                        _ => {
                            f.instruction(&Instruction::I32Store(ma));
                        }
                    },
                    ValType::I64 => {
                        f.instruction(&Instruction::I64Store(ma));
                    }
                    ValType::F32 => {
                        f.instruction(&Instruction::F32Store(ma));
                    }
                    ValType::F64 => {
                        f.instruction(&Instruction::F64Store(ma));
                    }
                    _ => {
                        f.instruction(&Instruction::I32Store(ma));
                    }
                };
            }

            // === LoadGlobal ===
            IrInstruction::LoadGlobal {
                dest, global_id, ..
            } => {
                let gidx = self
                    .ctx
                    .ir_global_to_idx
                    .get(global_id)
                    .copied()
                    .unwrap_or(0);
                f.instruction(&Instruction::GlobalGet(gidx));
                self.set_reg(f, *dest);
            }

            // === StoreGlobal ===
            IrInstruction::StoreGlobal { global_id, value } => {
                self.get_reg(f, *value);
                let gidx = self
                    .ctx
                    .ir_global_to_idx
                    .get(global_id)
                    .copied()
                    .unwrap_or(0);
                f.instruction(&Instruction::GlobalSet(gidx));
            }

            // === GetElementPtr ===
            IrInstruction::GetElementPtr {
                dest,
                ptr,
                indices,
                ty,
                ..
            } => {
                // GEP: result = ptr + sum(index * elem_size)
                let elem_size: i32 = gep_pointee_size(ty);
                self.get_reg(f, *ptr);
                for idx_reg in indices {
                    self.get_reg(f, *idx_reg);
                    if elem_size != 1 {
                        f.instruction(&Instruction::I32Const(elem_size));
                        f.instruction(&Instruction::I32Mul);
                    }
                    f.instruction(&Instruction::I32Add);
                }
                self.set_reg(f, *dest);
            }

            // === PtrAdd ===
            IrInstruction::PtrAdd {
                dest,
                ptr,
                offset,
                ty,
            } => {
                // PtrAdd offsets are ELEMENT indices scaled by the pointee
                // size — Cranelift multiplies by type_size(pointee); the
                // raw I32Add here silently dropped the scale, so e.g. the
                // QTensor_fusedQkvMatmul wrapper's Ptr<I64> header stores
                // landed at byte offsets 0/1/2/3 instead of 0/8/16/24,
                // corrupting the HaxeArray header (wild pushes → OOB at
                // prefill). Use the SAME size rule as the GEP arm above so
                // the two stay consistent on this backend; Ptr<U8> users
                // (byte offsets) keep scale 1 and are unaffected.
                let elem_size: i32 = gep_pointee_size(ty);
                self.get_reg(f, *ptr);
                self.get_reg(f, *offset);
                if elem_size != 1 {
                    f.instruction(&Instruction::I32Const(elem_size));
                    f.instruction(&Instruction::I32Mul);
                }
                f.instruction(&Instruction::I32Add);
                self.set_reg(f, *dest);
            }

            // === Alloc (shadow stack bump) ===
            IrInstruction::Alloc { dest, ty, count } => {
                let base_size = wasm_alloc_size(ty);
                match count {
                    Some(count_reg) => {
                        f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                        self.get_reg(f, *count_reg);
                        f.instruction(&Instruction::I32Const(base_size));
                        f.instruction(&Instruction::I32Mul);
                        f.instruction(&Instruction::I32Sub);
                        f.instruction(&Instruction::I32Const(-8i32)); // align mask
                        f.instruction(&Instruction::I32And);
                        f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));
                        f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                        self.set_reg(f, *dest);
                    }
                    None => {
                        let aligned = (base_size + 7) & !7;
                        f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                        f.instruction(&Instruction::I32Const(aligned.max(8)));
                        f.instruction(&Instruction::I32Sub);
                        f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));
                        f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                        self.set_reg(f, *dest);
                    }
                }
            }

            // === Free — real reclamation via the size-prefix-header allocator ===
            // Mirrors native's `call free(ptr)`. `free` is linker-aliased to
            // rayzor_obj_free, which reads the 8-byte size header at ptr-8 and
            // returns the block to dlmalloc. Worker-safe: no shared state, no FFI.
            // InsertFreePass emits Free only for non-escaping, dominating,
            // entry-block malloc-class objects, so `ptr` is provably dead.
            // Gated by RAYZOR_WASM_FREE (free_idx is None when disabled/unlinked).
            IrInstruction::Free { ptr } => {
                if self.ctx.wasm_free_enabled {
                    if let Some(idx) = self.ctx.free_idx {
                        self.get_reg(f, *ptr);
                        f.instruction(&Instruction::Call(idx));
                    }
                }
            }

            // === CallDirect ===
            IrInstruction::CallDirect {
                dest,
                func_id,
                args,
                ..
            } => {
                // Load args with type coercion to match callee signature
                let callee_params = self.ctx.func_param_types.get(func_id);
                let has_implicit_env = self.ctx.functions_with_env.contains(func_id);
                let param_offset = if has_implicit_env { 1 } else { 0 };
                if has_implicit_env {
                    f.instruction(&Instruction::I32Const(0));
                }
                for (i, arg) in args.iter().enumerate() {
                    self.get_reg(f, *arg);
                    let arg_ty = self.reg_wasm_type(*arg);
                    if let Some(params) = callee_params {
                        if let Some(&expected) = params.get(i + param_offset) as Option<&ValType> {
                            self.emit_type_coerce(f, arg_ty, expected);
                        }
                    }
                }
                let resolved_idx = self
                    .ctx
                    .ir_func_to_idx
                    .get(func_id)
                    .copied()
                    .or_else(|| self.ctx.func_id_fallback.get(func_id).copied())
                    // Last resort: look up by function name from any module
                    .or_else(|| {
                        for m in self.ctx.all_module_funcs.iter() {
                            if let Some(name) = m.get(func_id) {
                                if let Some(&idx) = self.ctx.import_name_to_idx.get(name) {
                                    return Some(idx);
                                }
                                if let Some(&idx) = self.ctx.func_name_to_idx.get(name) {
                                    return Some(idx);
                                }
                                // Try qualified_to_import (the name might be a qualified Haxe name)
                                if let Some(&idx) = self.ctx.qualified_to_import.get(name) {
                                    return Some(idx);
                                }
                                // Try bare method name match against imports
                                let bare = name.rsplit('.').next().unwrap_or(name);
                                let bare_underscore = name.rsplit('_').next().unwrap_or(name);
                                for (imp_name, &idx) in &self.ctx.import_name_to_idx {
                                    if imp_name.ends_with(bare)
                                        || imp_name.ends_with(bare_underscore)
                                    {
                                        return Some(idx);
                                    }
                                }
                            }
                        }
                        None
                    });
                if let Some(idx) = resolved_idx {
                    // Check if the callee expects a different number of params.
                    // If so, adjust the stack (drop extra or push zeros).
                    // Get the callee's actual WASM signature from the import/function type
                    let callee_sig = if (idx as usize) < self.ctx.imports.len() {
                        let imp = &self.ctx.imports[idx as usize];
                        self.ctx.types.get(imp.type_idx as usize).cloned()
                    } else {
                        self.ctx
                            .internals
                            .iter()
                            .find(|internal| internal.func_idx == idx)
                            .and_then(|internal| self.ctx.types.get(internal.type_idx as usize))
                            .cloned()
                    };
                    if let Some((expected_params, expected_results)) = &callee_sig {
                        let pushed = args.len() + param_offset;
                        // Adjust params: drop extra or push zeros
                        if pushed > expected_params.len() {
                            for _ in 0..(pushed - expected_params.len()) {
                                f.instruction(&Instruction::Drop);
                            }
                        } else if pushed < expected_params.len() {
                            for i in pushed..expected_params.len() {
                                emit_zero(f, expected_params[i]);
                            }
                        }
                    }
                    f.instruction(&Instruction::Call(idx));
                    // Adjust return value mismatch
                    if let Some((_, expected_results)) = &callee_sig {
                        if dest.is_some() && expected_results.is_empty() {
                            // Callee returns void but caller wants a value — push zero
                            emit_zero(f, self.reg_wasm_type(dest.unwrap()));
                        } else if dest.is_none() && !expected_results.is_empty() {
                            // Callee returns a value but caller doesn't want it — drop
                            f.instruction(&Instruction::Drop);
                        }
                    }
                } else {
                    // Unknown function — try to find its name for diagnostics
                    let mut found_name = None;
                    for m in self.ctx.all_module_funcs.iter() {
                        if let Some(name) = m.get(func_id) {
                            found_name = Some(name.clone());
                            break;
                        }
                    }
                    for _ in args {
                        f.instruction(&Instruction::Drop);
                    }
                    f.instruction(&Instruction::Unreachable);
                }
                if let Some(d) = dest {
                    self.set_reg(f, *d);
                }
            }

            // === CallIndirect ===
            IrInstruction::CallIndirect {
                dest,
                func_ptr,
                args,
                signature,
                ..
            } => {
                let actual_user_params: Vec<ValType> =
                    args.iter().map(|a| self.reg_wasm_type(*a)).collect();
                let actual_results = dest
                    .map(|d| vec![self.reg_wasm_type(d)])
                    .unwrap_or_default();
                let (user_params, mut results) = match signature {
                    IrType::Function {
                        params,
                        return_type,
                        ..
                    } => {
                        let p: Vec<ValType> = params.iter().map(ir_type_to_wasm).collect();
                        let r = if matches!(return_type.as_ref(), IrType::Void) {
                            vec![]
                        } else {
                            vec![ir_type_to_wasm(return_type)]
                        };
                        (p, r)
                    }
                    _ => (actual_user_params.clone(), actual_results.clone()),
                };
                // The call_indirect type must equal the callee's DEFINED
                // type. Vtable dispatch thunks are defined from the same
                // method signature this site's MIR signature carries, so
                // trusting the signature matches them by construction —
                // including Void methods (no result). The legacy override
                // below (force the dest register's type when present)
                // breaks Void-method dispatch: site expects [I32], thunk
                // is (). RAYZOR_WASM_LEGACY_CI_RESULTS=1 restores it.
                let trust_signature = std::env::var_os("RAYZOR_WASM_LEGACY_CI_RESULTS").is_none();
                if !trust_signature && dest.is_some() && results != actual_results {
                    results = actual_results.clone();
                }

                let mut params = Vec::with_capacity(user_params.len() + 1);
                params.push(ValType::I32);
                params.extend(user_params.iter().copied());
                let primary_hit = self
                    .ctx
                    .type_map
                    .get(&(params.clone(), results.clone()))
                    .copied();
                let type_idx = primary_hit
                    .or_else(|| {
                        let mut actual_params = Vec::with_capacity(actual_user_params.len() + 1);
                        actual_params.push(ValType::I32);
                        actual_params.extend(actual_user_params.iter().copied());
                        self.ctx
                            .type_map
                            .get(&(actual_params, actual_results.clone()))
                            .copied()
                    })
                    .unwrap_or(0);
                if std::env::var_os("RAYZOR_WASM_CI_DEBUG").is_some() {
                    eprintln!(
                        "[wasm-ci] sig_params={:?} results={:?} actual={:?} primary_hit={:?} type_idx={}",
                        params, results, actual_user_params, primary_hit, type_idx
                    );
                }

                // Stack order for call_indirect is: params..., table_index.
                // Closure layout is { table_slot: i32, env_ptr: i32 } at offsets 0/8.
                self.get_reg(f, *func_ptr);
                f.instruction(&Instruction::I32Load(MemArg {
                    offset: 8,
                    align: 2,
                    memory_index: 0,
                }));
                for (i, arg) in args.iter().enumerate() {
                    self.get_reg(f, *arg);
                    let arg_ty = self.reg_wasm_type(*arg);
                    if let Some(&expected) = params.get(i + 1) {
                        self.emit_type_coerce(f, arg_ty, expected);
                    }
                }
                self.get_reg(f, *func_ptr);
                f.instruction(&Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }));
                f.instruction(&Instruction::CallIndirect {
                    type_index: type_idx,
                    table_index: 0,
                });
                match (dest, results.first()) {
                    (Some(d), Some(&res_ty)) => {
                        // Coerce the declared result to the dest register's
                        // type when they disagree (e.g. i32 result into an
                        // f64 register).
                        let dest_ty = self.reg_wasm_type(*d);
                        self.emit_type_coerce(f, res_ty, dest_ty);
                        self.set_reg(f, *d);
                    }
                    (Some(d), None) => {
                        // Void callee but the MIR call kept a (dummy) dest —
                        // build_call_indirect always mints one. Zero it.
                        emit_zero(f, self.reg_wasm_type(*d));
                        self.set_reg(f, *d);
                    }
                    (None, Some(_)) => {
                        f.instruction(&Instruction::Drop);
                    }
                    (None, None) => {}
                }
            }

            // === Select (ternary) ===
            IrInstruction::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => {
                self.get_reg(f, *true_val);
                self.get_reg(f, *false_val);
                self.get_reg(f, *condition);
                f.instruction(&Instruction::Select);
                self.set_reg(f, *dest);
            }

            // === Phi (in-block form) ===
            IrInstruction::Phi { dest, incoming } => {
                if let Some((val, _)) = incoming.first() {
                    self.get_reg(f, *val);
                    self.set_reg(f, *dest);
                }
            }

            // === FunctionRef ===
            IrInstruction::FunctionRef { dest, func_id } => {
                let idx = self.ctx.ir_func_to_idx.get(func_id).copied().unwrap_or(0);
                // Function values use the same closure-object ABI as lambdas:
                // { table_slot, env_ptr }. Static function references carry a
                // null env pointer; indirect calls still pass that env lane.
                f.instruction(&Instruction::I32Const(16));
                if let Some(&malloc_idx) = self.ctx.import_name_to_idx.get("malloc") {
                    f.instruction(&Instruction::Call(malloc_idx));
                } else if let Some(&malloc_idx) = self.ctx.func_name_to_idx.get("malloc") {
                    f.instruction(&Instruction::Call(malloc_idx));
                } else {
                    f.instruction(&Instruction::Drop);
                    f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                    f.instruction(&Instruction::I32Const(16));
                    f.instruction(&Instruction::I32Sub);
                    f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));
                    f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                }
                self.set_reg(f, *dest);

                f.instruction(&Instruction::I32Const(idx as i32));
                f.instruction(&Instruction::RefFunc(idx));
                f.instruction(&Instruction::TableSet(0));

                self.get_reg(f, *dest);
                f.instruction(&Instruction::I32Const(idx as i32));
                f.instruction(&Instruction::I32Store(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }));

                self.get_reg(f, *dest);
                f.instruction(&Instruction::I32Const(0));
                f.instruction(&Instruction::I32Store(MemArg {
                    offset: 8,
                    align: 2,
                    memory_index: 0,
                }));
            }

            // === MakeClosure ===
            IrInstruction::MakeClosure {
                dest,
                func_id,
                captured_values,
            } => {
                // Layout matches native Cranelift backend: { fn_ptr, env_ptr } 16 bytes,
                // with fn_idx at offset 0 and env_ptr at offset 8. The environment struct
                // is allocated separately and holds captured values at 8-byte strides.
                // This matches Thread_spawn / Future_create MIR wrappers which load the
                // env_ptr from `*(closure + 8)`.

                let fn_idx = self.ctx.ir_func_to_idx.get(func_id).copied().unwrap_or(0);

                // 1. Allocate closure struct (16 bytes).
                f.instruction(&Instruction::I32Const(16));
                if let Some(&malloc_idx) = self.ctx.import_name_to_idx.get("malloc") {
                    f.instruction(&Instruction::Call(malloc_idx));
                } else if let Some(&malloc_idx) = self.ctx.func_name_to_idx.get("malloc") {
                    f.instruction(&Instruction::Call(malloc_idx));
                } else {
                    f.instruction(&Instruction::Drop);
                    f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                    f.instruction(&Instruction::I32Const(16));
                    f.instruction(&Instruction::I32Sub);
                    f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));
                    f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                }
                self.set_reg(f, *dest);

                // 2. Ensure the indirect function table has our fn_idx slot populated.
                // `ref.func` is what the linker remaps during module merging, so emitting
                // `table.set(fn_idx, ref.func(fn_idx))` lets subsequent table.get calls
                // resolve to the correct post-link function.
                f.instruction(&Instruction::I32Const(fn_idx as i32));
                f.instruction(&Instruction::RefFunc(fn_idx));
                f.instruction(&Instruction::TableSet(0));

                // 3. Store fn_idx at closure offset 0.
                self.get_reg(f, *dest);
                f.instruction(&Instruction::I32Const(fn_idx as i32));
                f.instruction(&Instruction::I32Store(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }));

                // 4. Allocate environment and store env_ptr at closure offset 8.
                //    We leave the closure pointer on the WASM stack, malloc the env,
                //    then i32.store the env pointer directly at closure+8.
                if !captured_values.is_empty() {
                    let env_size = (captured_values.len() * 8) as i32;
                    // Push closure pointer (for the env_ptr store).
                    self.get_reg(f, *dest);
                    // Call malloc(env_size) — result (env_ptr) is on stack.
                    f.instruction(&Instruction::I32Const(env_size));
                    if let Some(&malloc_idx) = self.ctx.import_name_to_idx.get("malloc") {
                        f.instruction(&Instruction::Call(malloc_idx));
                    } else if let Some(&malloc_idx) = self.ctx.func_name_to_idx.get("malloc") {
                        f.instruction(&Instruction::Call(malloc_idx));
                    } else {
                        f.instruction(&Instruction::Drop);
                        f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                        f.instruction(&Instruction::I32Const(env_size));
                        f.instruction(&Instruction::I32Sub);
                        f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));
                        f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                    }
                    // Stack: [closure_ptr, env_ptr] — store env_ptr at closure+8.
                    f.instruction(&Instruction::I32Store(MemArg {
                        offset: 8,
                        align: 2,
                        memory_index: 0,
                    }));

                    // Now store each capture in env by reloading env_ptr from closure+8.
                    for (i, cap) in captured_values.iter().enumerate() {
                        self.get_reg(f, *dest);
                        f.instruction(&Instruction::I32Load(MemArg {
                            offset: 8,
                            align: 2,
                            memory_index: 0,
                        }));
                        self.get_reg(f, *cap);
                        f.instruction(&Instruction::I32Store(MemArg {
                            offset: (i * 8) as u64,
                            align: 2,
                            memory_index: 0,
                        }));
                    }
                } else {
                    // No captures: store null env_ptr at offset 8.
                    self.get_reg(f, *dest);
                    f.instruction(&Instruction::I32Const(0));
                    f.instruction(&Instruction::I32Store(MemArg {
                        offset: 8,
                        align: 2,
                        memory_index: 0,
                    }));
                }
            }

            // === ClosureFunc ===
            IrInstruction::ClosureFunc { dest, closure } => {
                self.get_reg(f, *closure);
                f.instruction(&Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }));
                self.set_reg(f, *dest);
            }

            // === ClosureEnv ===
            IrInstruction::ClosureEnv { dest, closure } => {
                // Env is a separately-allocated struct pointer stored at offset 8.
                self.get_reg(f, *closure);
                f.instruction(&Instruction::I32Load(MemArg {
                    offset: 8,
                    align: 2,
                    memory_index: 0,
                }));
                self.set_reg(f, *dest);
            }

            // === MemCopy ===
            IrInstruction::MemCopy { dest, src, size } => {
                self.get_reg(f, *dest);
                self.get_reg(f, *src);
                self.get_reg(f, *size);
                f.instruction(&Instruction::MemoryCopy {
                    src_mem: 0,
                    dst_mem: 0,
                });
            }

            // === MemSet ===
            IrInstruction::MemSet { dest, value, size } => {
                self.get_reg(f, *dest);
                self.get_reg(f, *value);
                self.get_reg(f, *size);
                f.instruction(&Instruction::MemoryFill(0));
            }

            // === Undef ===
            IrInstruction::Undef { dest, .. } => {
                // Use the local's actual WASM type, not the IR type, to avoid
                // mismatches where F32/F64 register is allocated as I32 local.
                emit_zero(f, self.reg_wasm_type(*dest));
                self.set_reg(f, *dest);
            }

            // === Return (as instruction) ===
            IrInstruction::Return { value } => {
                if let Some(v) = value {
                    self.get_reg(f, *v);
                    if matches!(self.ir_func.signature.return_type, IrType::Void) {
                        f.instruction(&Instruction::Drop);
                    } else {
                        let actual_ty = self
                            .local_type_of(*v)
                            .unwrap_or_else(|| self.reg_wasm_type(*v));
                        let return_ty = ir_type_to_wasm(&self.ir_func.signature.return_type);
                        emit_bitcast(f, actual_ty, return_ty);
                    }
                } else if !matches!(self.ir_func.signature.return_type, IrType::Void) {
                    emit_zero(f, ir_type_to_wasm(&self.ir_func.signature.return_type));
                }
                self.emit_sp_restore(f);
                f.instruction(&Instruction::Return);
            }

            // === ExtractValue ===
            IrInstruction::ExtractValue {
                dest,
                aggregate,
                indices,
            } => {
                self.get_reg(f, *aggregate);
                if let Some(&idx) = indices.first() {
                    f.instruction(&Instruction::I32Const(idx as i32 * 8));
                    f.instruction(&Instruction::I32Add);
                }
                emit_typed_load(f, self.reg_wasm_type(*dest));
                self.set_reg(f, *dest);
            }

            // === InsertValue ===
            IrInstruction::InsertValue {
                dest,
                aggregate,
                value,
                indices,
            } => {
                // Copy aggregate pointer to dest.
                self.get_reg(f, *aggregate);
                self.set_reg(f, *dest);
                // Compute target address.
                self.get_reg(f, *dest);
                if let Some(&idx) = indices.first() {
                    f.instruction(&Instruction::I32Const(idx as i32 * 8));
                    f.instruction(&Instruction::I32Add);
                }
                self.get_reg(f, *value);
                emit_typed_store(f, self.reg_wasm_type(*value));
            }

            // === CreateStruct ===
            IrInstruction::CreateStruct { dest, fields, .. } => {
                let alloc_size = ((fields.len() * 8 + 7) & !7) as i32;
                f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                f.instruction(&Instruction::I32Const(alloc_size.max(8)));
                f.instruction(&Instruction::I32Sub);
                f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));
                f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                self.set_reg(f, *dest);
                for (i, field_reg) in fields.iter().enumerate() {
                    self.get_reg(f, *dest);
                    self.get_reg(f, *field_reg);
                    emit_typed_store_offset(f, self.reg_wasm_type(*field_reg), (i * 8) as u64);
                }
            }

            // === CreateUnion ===
            IrInstruction::CreateUnion {
                dest,
                discriminant,
                value,
                ..
            } => {
                // [discriminant: i32 @ 0] [value @ 8]
                f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                f.instruction(&Instruction::I32Const(16));
                f.instruction(&Instruction::I32Sub);
                f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));
                f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                self.set_reg(f, *dest);
                // Store discriminant.
                self.get_reg(f, *dest);
                f.instruction(&Instruction::I32Const(*discriminant as i32));
                f.instruction(&Instruction::I32Store(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }));
                // Store value at offset 8.
                self.get_reg(f, *dest);
                self.get_reg(f, *value);
                emit_typed_store_offset(f, self.reg_wasm_type(*value), 8);
            }

            // === ExtractDiscriminant ===
            IrInstruction::ExtractDiscriminant { dest, union_val } => {
                self.get_reg(f, *union_val);
                f.instruction(&Instruction::I32Load(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }));
                self.set_reg(f, *dest);
            }

            // === ExtractUnionValue ===
            IrInstruction::ExtractUnionValue {
                dest, union_val, ..
            } => {
                self.get_reg(f, *union_val);
                emit_typed_load_offset(f, self.reg_wasm_type(*dest), 8);
                self.set_reg(f, *dest);
            }

            // === LandingPad (no exceptions in Phase 1) ===
            IrInstruction::LandingPad { dest, .. } => {
                f.instruction(&Instruction::I32Const(0));
                self.set_reg(f, *dest);
            }

            // === Throw / Resume / Panic ===
            // Note: Haxe `throw` is lowered to a direct call to
            // `rayzor_throw_typed` (which prints + traps), not to this
            // IrInstruction::Throw, so the message surfaces there.
            IrInstruction::Throw { .. }
            | IrInstruction::Resume { .. }
            | IrInstruction::Panic { .. } => {
                f.instruction(&Instruction::Unreachable);
            }

            // === No-ops ===
            IrInstruction::EndBorrow { .. }
            | IrInstruction::DebugLoc { .. }
            | IrInstruction::Jump { .. }
            | IrInstruction::Branch { .. }
            | IrInstruction::Switch { .. } => {}

            // === InlineAsm (still unsupported) ===
            IrInstruction::InlineAsm { .. } => {
                f.instruction(&Instruction::Unreachable);
            }

            // === SIMD vector instructions → WASM SIMD128 ===
            // Currently only f32x4 (16 bytes, 4 lanes) is exercised by SIMD4f.
            // The lowering pattern: get_reg(operands) → emit f32x4.* → set_reg(dest).
            // Locals/params with IrType::Vector{F32,4} are typed as ValType::V128
            // via ir_type_to_wasm.

            // Splat: scalar f32 → vec
            IrInstruction::VectorSplat { dest, scalar, .. } => {
                self.get_reg(f, *scalar);
                f.instruction(&Instruction::F32x4Splat);
                self.set_reg(f, *dest);
            }

            // Element-wise binary ops (add/sub/mul/div)
            IrInstruction::VectorBinOp {
                dest,
                op,
                left,
                right,
                ..
            } => {
                self.get_reg(f, *left);
                self.get_reg(f, *right);
                match op {
                    BinaryOp::Add | BinaryOp::FAdd => {
                        f.instruction(&Instruction::F32x4Add);
                    }
                    BinaryOp::Sub | BinaryOp::FSub => {
                        f.instruction(&Instruction::F32x4Sub);
                    }
                    BinaryOp::Mul | BinaryOp::FMul => {
                        f.instruction(&Instruction::F32x4Mul);
                    }
                    BinaryOp::Div | BinaryOp::FDiv => {
                        f.instruction(&Instruction::F32x4Div);
                    }
                    _ => {
                        // Unsupported vector binop (And/Or/Xor/Shl on f32 not meaningful)
                        f.instruction(&Instruction::Unreachable);
                    }
                }
                self.set_reg(f, *dest);
            }

            // Extract a single lane (compile-time index)
            IrInstruction::VectorExtract {
                dest,
                vector,
                index,
            } => {
                self.get_reg(f, *vector);
                f.instruction(&Instruction::F32x4ExtractLane(*index));
                self.set_reg(f, *dest);
            }

            // Insert a single lane (compile-time index)
            IrInstruction::VectorInsert {
                dest,
                vector,
                scalar,
                index,
            } => {
                self.get_reg(f, *vector);
                self.get_reg(f, *scalar);
                f.instruction(&Instruction::F32x4ReplaceLane(*index));
                self.set_reg(f, *dest);
            }

            // Element-wise unary ops
            IrInstruction::VectorUnaryOp {
                dest, op, operand, ..
            } => {
                self.get_reg(f, *operand);
                match op {
                    VectorUnaryOpKind::Sqrt => {
                        f.instruction(&Instruction::F32x4Sqrt);
                    }
                    VectorUnaryOpKind::Abs => {
                        f.instruction(&Instruction::F32x4Abs);
                    }
                    VectorUnaryOpKind::Neg => {
                        f.instruction(&Instruction::F32x4Neg);
                    }
                    VectorUnaryOpKind::Ceil => {
                        f.instruction(&Instruction::F32x4Ceil);
                    }
                    VectorUnaryOpKind::Floor => {
                        f.instruction(&Instruction::F32x4Floor);
                    }
                    VectorUnaryOpKind::Trunc => {
                        f.instruction(&Instruction::F32x4Trunc);
                    }
                    VectorUnaryOpKind::Round => {
                        f.instruction(&Instruction::F32x4Nearest);
                    }
                }
                self.set_reg(f, *dest);
            }

            // Element-wise min/max
            IrInstruction::VectorMinMax {
                dest,
                op,
                left,
                right,
                ..
            } => {
                self.get_reg(f, *left);
                self.get_reg(f, *right);
                match op {
                    VectorMinMaxKind::Min => {
                        f.instruction(&Instruction::F32x4Min);
                    }
                    VectorMinMaxKind::Max => {
                        f.instruction(&Instruction::F32x4Max);
                    }
                }
                self.set_reg(f, *dest);
            }

            // Horizontal reduction. WASM has no native f32x4 horizontal reduce,
            // so we extract all 4 lanes and sum them.
            IrInstruction::VectorReduce { dest, op, vector } => {
                match op {
                    BinaryOp::Add | BinaryOp::FAdd => {
                        // s = lane0 + lane1 + lane2 + lane3
                        self.get_reg(f, *vector);
                        f.instruction(&Instruction::F32x4ExtractLane(0));
                        self.get_reg(f, *vector);
                        f.instruction(&Instruction::F32x4ExtractLane(1));
                        f.instruction(&Instruction::F32Add);
                        self.get_reg(f, *vector);
                        f.instruction(&Instruction::F32x4ExtractLane(2));
                        f.instruction(&Instruction::F32Add);
                        self.get_reg(f, *vector);
                        f.instruction(&Instruction::F32x4ExtractLane(3));
                        f.instruction(&Instruction::F32Add);
                    }
                    BinaryOp::Mul | BinaryOp::FMul => {
                        self.get_reg(f, *vector);
                        f.instruction(&Instruction::F32x4ExtractLane(0));
                        self.get_reg(f, *vector);
                        f.instruction(&Instruction::F32x4ExtractLane(1));
                        f.instruction(&Instruction::F32Mul);
                        self.get_reg(f, *vector);
                        f.instruction(&Instruction::F32x4ExtractLane(2));
                        f.instruction(&Instruction::F32Mul);
                        self.get_reg(f, *vector);
                        f.instruction(&Instruction::F32x4ExtractLane(3));
                        f.instruction(&Instruction::F32Mul);
                    }
                    _ => {
                        f.instruction(&Instruction::Unreachable);
                    }
                }
                self.set_reg(f, *dest);
            }

            // Load 16 bytes from memory into a v128
            IrInstruction::VectorLoad { dest, ptr, .. } => {
                self.get_reg(f, *ptr);
                let ma = MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                };
                f.instruction(&Instruction::V128Load(ma));
                self.set_reg(f, *dest);
            }

            // Store v128 to 16 bytes of memory
            IrInstruction::VectorStore { ptr, value, .. } => {
                self.get_reg(f, *ptr);
                self.get_reg(f, *value);
                let ma = MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                };
                f.instruction(&Instruction::V128Store(ma));
            }

            // === Atomic operations (sequentially consistent) ===
            // align: 2 == log2(4); atomics require a naturally-aligned MemArg.
            IrInstruction::AtomicLoad { dest, ptr, .. } => {
                self.get_reg(f, *ptr);
                let ma = MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                };
                f.instruction(&Instruction::I32AtomicLoad(ma));
                self.set_reg(f, *dest);
            }
            IrInstruction::AtomicStore { ptr, value, .. } => {
                self.get_reg(f, *ptr);
                self.get_reg(f, *value);
                let ma = MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                };
                f.instruction(&Instruction::I32AtomicStore(ma));
            }
            IrInstruction::AtomicRmw {
                dest,
                op,
                ptr,
                value,
                ..
            } => {
                self.get_reg(f, *ptr);
                self.get_reg(f, *value);
                let ma = MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                };
                match op {
                    AtomicRmwOp::Add => f.instruction(&Instruction::I32AtomicRmwAdd(ma)),
                    AtomicRmwOp::Sub => f.instruction(&Instruction::I32AtomicRmwSub(ma)),
                    AtomicRmwOp::And => f.instruction(&Instruction::I32AtomicRmwAnd(ma)),
                    AtomicRmwOp::Or => f.instruction(&Instruction::I32AtomicRmwOr(ma)),
                    AtomicRmwOp::Xor => f.instruction(&Instruction::I32AtomicRmwXor(ma)),
                    AtomicRmwOp::Xchg => f.instruction(&Instruction::I32AtomicRmwXchg(ma)),
                };
                self.set_reg(f, *dest);
            }
            IrInstruction::AtomicCas {
                dest,
                ptr,
                expected,
                replacement,
                ..
            } => {
                // wasm stack order: addr, expected, replacement
                self.get_reg(f, *ptr);
                self.get_reg(f, *expected);
                self.get_reg(f, *replacement);
                let ma = MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                };
                f.instruction(&Instruction::I32AtomicRmwCmpxchg(ma));
                self.set_reg(f, *dest);
            }

            // Catch-all for anything else.
            _ => {
                f.instruction(&Instruction::Unreachable);
            }
        }
    }

    // ------------------------------------------------------------------
    // Constant emission
    // ------------------------------------------------------------------

    fn emit_const(&self, f: &mut Function, value: &IrValue) {
        match value {
            IrValue::Void | IrValue::Undef | IrValue::Null => {
                f.instruction(&Instruction::I32Const(0));
            }
            IrValue::Bool(b) => {
                f.instruction(&Instruction::I32Const(if *b { 1 } else { 0 }));
            }
            IrValue::I8(v) => {
                f.instruction(&Instruction::I32Const(*v as i32));
            }
            IrValue::I16(v) => {
                f.instruction(&Instruction::I32Const(*v as i32));
            }
            IrValue::I32(v) => {
                f.instruction(&Instruction::I32Const(*v));
            }
            IrValue::I64(v) => {
                f.instruction(&Instruction::I32Const(*v as i32));
            }
            IrValue::U8(v) => {
                f.instruction(&Instruction::I32Const(*v as i32));
            }
            IrValue::U16(v) => {
                f.instruction(&Instruction::I32Const(*v as i32));
            }
            IrValue::U32(v) => {
                f.instruction(&Instruction::I32Const(*v as i32));
            }
            IrValue::U64(v) => {
                f.instruction(&Instruction::I32Const(*v as i32));
            }
            IrValue::F32(v) => {
                f.instruction(&Instruction::F32Const(Ieee32::from(*v)));
            }
            IrValue::F64(v) => {
                f.instruction(&Instruction::F64Const(Ieee64::from(*v)));
            }
            IrValue::String(s) => {
                let off = self
                    .ctx
                    .string_offsets
                    .get(s.as_str())
                    .copied()
                    .unwrap_or(0);
                f.instruction(&Instruction::I32Const(off as i32));
            }
            IrValue::Function(fid) => {
                let idx = self.ctx.ir_func_to_idx.get(fid).copied().unwrap_or(0);
                f.instruction(&Instruction::I32Const(idx as i32));
            }
            IrValue::Closure { function, .. } => {
                let idx = self.ctx.ir_func_to_idx.get(function).copied().unwrap_or(0);
                f.instruction(&Instruction::I32Const(idx as i32));
            }
            IrValue::Array(_) | IrValue::Struct(_) => {
                f.instruction(&Instruction::I32Const(0));
            }
        }
    }

    // ------------------------------------------------------------------
    // BinOp emission
    // ------------------------------------------------------------------

    fn emit_binop(&self, f: &mut Function, op: BinaryOp, dest_ty: ValType) {
        match (op, dest_ty) {
            // When MIR uses Add/Sub/Mul/Div on floats (not FAdd/FSub/FMul/FDiv),
            // we detect via operand type and emit the float variant.
            (BinaryOp::Add, ValType::F64) | (BinaryOp::FAdd, ValType::F64) => {
                f.instruction(&Instruction::F64Add);
            }
            (BinaryOp::Sub, ValType::F64) | (BinaryOp::FSub, ValType::F64) => {
                f.instruction(&Instruction::F64Sub);
            }
            (BinaryOp::Mul, ValType::F64) | (BinaryOp::FMul, ValType::F64) => {
                f.instruction(&Instruction::F64Mul);
            }
            (BinaryOp::Div, ValType::F64) | (BinaryOp::FDiv, ValType::F64) => {
                f.instruction(&Instruction::F64Div);
            }
            (BinaryOp::Add, ValType::F32) | (BinaryOp::FAdd, ValType::F32) => {
                f.instruction(&Instruction::F32Add);
            }
            (BinaryOp::Sub, ValType::F32) | (BinaryOp::FSub, ValType::F32) => {
                f.instruction(&Instruction::F32Sub);
            }
            (BinaryOp::Mul, ValType::F32) | (BinaryOp::FMul, ValType::F32) => {
                f.instruction(&Instruction::F32Mul);
            }
            (BinaryOp::Div, ValType::F32) | (BinaryOp::FDiv, ValType::F32) => {
                f.instruction(&Instruction::F32Div);
            }

            (BinaryOp::Add, ValType::I32) => {
                f.instruction(&Instruction::I32Add);
            }
            (BinaryOp::Sub, ValType::I32) => {
                f.instruction(&Instruction::I32Sub);
            }
            (BinaryOp::Mul, ValType::I32) => {
                f.instruction(&Instruction::I32Mul);
            }
            (BinaryOp::Div, ValType::I32) => {
                f.instruction(&Instruction::I32DivS);
            }
            (BinaryOp::Rem, ValType::I32) => {
                f.instruction(&Instruction::I32RemS);
            }
            (BinaryOp::And, ValType::I32) => {
                f.instruction(&Instruction::I32And);
            }
            (BinaryOp::Or, ValType::I32) => {
                f.instruction(&Instruction::I32Or);
            }
            (BinaryOp::Xor, ValType::I32) => {
                f.instruction(&Instruction::I32Xor);
            }
            (BinaryOp::Shl, ValType::I32) => {
                f.instruction(&Instruction::I32Shl);
            }
            (BinaryOp::Shr, ValType::I32) => {
                f.instruction(&Instruction::I32ShrS);
            }
            (BinaryOp::Ushr, ValType::I32) => {
                f.instruction(&Instruction::I32ShrU);
            }

            (BinaryOp::Add, ValType::I64) => {
                f.instruction(&Instruction::I64Add);
            }
            (BinaryOp::Sub, ValType::I64) => {
                f.instruction(&Instruction::I64Sub);
            }
            (BinaryOp::Mul, ValType::I64) => {
                f.instruction(&Instruction::I64Mul);
            }
            (BinaryOp::Div, ValType::I64) => {
                f.instruction(&Instruction::I64DivS);
            }
            (BinaryOp::Rem, ValType::I64) => {
                f.instruction(&Instruction::I64RemS);
            }
            (BinaryOp::And, ValType::I64) => {
                f.instruction(&Instruction::I64And);
            }
            (BinaryOp::Or, ValType::I64) => {
                f.instruction(&Instruction::I64Or);
            }
            (BinaryOp::Xor, ValType::I64) => {
                f.instruction(&Instruction::I64Xor);
            }
            (BinaryOp::Shl, ValType::I64) => {
                f.instruction(&Instruction::I64Shl);
            }
            (BinaryOp::Shr, ValType::I64) => {
                f.instruction(&Instruction::I64ShrS);
            }
            (BinaryOp::Ushr, ValType::I64) => {
                f.instruction(&Instruction::I64ShrU);
            }

            (BinaryOp::FAdd, ValType::F32) => {
                f.instruction(&Instruction::F32Add);
            }
            (BinaryOp::FSub, ValType::F32) => {
                f.instruction(&Instruction::F32Sub);
            }
            (BinaryOp::FMul, ValType::F32) => {
                f.instruction(&Instruction::F32Mul);
            }
            (BinaryOp::FDiv, ValType::F32) => {
                f.instruction(&Instruction::F32Div);
            }

            (BinaryOp::FAdd, ValType::F64) => {
                f.instruction(&Instruction::F64Add);
            }
            (BinaryOp::FSub, ValType::F64) => {
                f.instruction(&Instruction::F64Sub);
            }
            (BinaryOp::FMul, ValType::F64) => {
                f.instruction(&Instruction::F64Mul);
            }
            (BinaryOp::FDiv, ValType::F64) => {
                f.instruction(&Instruction::F64Div);
            }

            // FRem has no direct WASM equivalent.
            (BinaryOp::FRem, _) => {
                f.instruction(&Instruction::Unreachable);
            }

            // Fallback: default to i32 ops for pointers etc.
            (BinaryOp::Add, _) => {
                f.instruction(&Instruction::I32Add);
            }
            (BinaryOp::Sub, _) => {
                f.instruction(&Instruction::I32Sub);
            }
            (BinaryOp::Mul, _) => {
                f.instruction(&Instruction::I32Mul);
            }
            (BinaryOp::Div, _) => {
                f.instruction(&Instruction::I32DivS);
            }
            (BinaryOp::Rem, _) => {
                f.instruction(&Instruction::I32RemS);
            }
            (BinaryOp::And, _) => {
                f.instruction(&Instruction::I32And);
            }
            (BinaryOp::Or, _) => {
                f.instruction(&Instruction::I32Or);
            }
            (BinaryOp::Xor, _) => {
                f.instruction(&Instruction::I32Xor);
            }
            (BinaryOp::Shl, _) => {
                f.instruction(&Instruction::I32Shl);
            }
            (BinaryOp::Shr, _) => {
                f.instruction(&Instruction::I32ShrS);
            }
            (BinaryOp::Ushr, _) => {
                f.instruction(&Instruction::I32ShrU);
            }
            (BinaryOp::FAdd, _) => {
                f.instruction(&Instruction::F64Add);
            }
            (BinaryOp::FSub, _) => {
                f.instruction(&Instruction::F64Sub);
            }
            (BinaryOp::FMul, _) => {
                f.instruction(&Instruction::F64Mul);
            }
            (BinaryOp::FDiv, _) => {
                f.instruction(&Instruction::F64Div);
            }
        }
    }

    // ------------------------------------------------------------------
    // UnOp emission
    // ------------------------------------------------------------------

    fn emit_unop(&self, f: &mut Function, op: UnaryOp, operand: IrId, dest_ty: ValType) {
        match (op, dest_ty) {
            (UnaryOp::Neg, ValType::F64) => {
                self.get_reg(f, operand);
                self.emit_type_coerce(f, self.reg_wasm_type(operand), ValType::F64);
                f.instruction(&Instruction::F64Neg);
            }
            (UnaryOp::Neg, ValType::F32) => {
                self.get_reg(f, operand);
                self.emit_type_coerce(f, self.reg_wasm_type(operand), ValType::F32);
                f.instruction(&Instruction::F32Neg);
            }
            (UnaryOp::Neg, ValType::I32) => {
                f.instruction(&Instruction::I32Const(0));
                self.get_reg(f, operand);
                f.instruction(&Instruction::I32Sub);
            }
            (UnaryOp::Neg, ValType::I64) => {
                f.instruction(&Instruction::I64Const(0));
                self.get_reg(f, operand);
                f.instruction(&Instruction::I64Sub);
            }
            (UnaryOp::Not, ValType::I32) => {
                self.get_reg(f, operand);
                f.instruction(&Instruction::I32Eqz);
            }
            (UnaryOp::Not, ValType::I64) => {
                self.get_reg(f, operand);
                f.instruction(&Instruction::I64Eqz);
            }
            (UnaryOp::FNeg, ValType::F32) => {
                self.get_reg(f, operand);
                f.instruction(&Instruction::F32Neg);
            }
            (UnaryOp::FNeg, ValType::F64) => {
                self.get_reg(f, operand);
                f.instruction(&Instruction::F64Neg);
            }
            (UnaryOp::Neg, _) => {
                f.instruction(&Instruction::I32Const(0));
                self.get_reg(f, operand);
                f.instruction(&Instruction::I32Sub);
            }
            (UnaryOp::Not, _) => {
                self.get_reg(f, operand);
                f.instruction(&Instruction::I32Eqz);
            }
            (UnaryOp::FNeg, _) => {
                self.get_reg(f, operand);
                f.instruction(&Instruction::F64Neg);
            }
        }
    }

    // ------------------------------------------------------------------
    // Cmp emission
    // ------------------------------------------------------------------

    fn emit_cmp(&self, f: &mut Function, op: CompareOp, operand_ty: ValType) {
        match (op, operand_ty) {
            // MIR uses Eq/Ne/Lt/Le/Gt/Ge for both int and float comparisons.
            // When operand type is F64/F32, emit float comparison instructions.
            (CompareOp::Eq, ValType::F64) | (CompareOp::FEq, ValType::F64) => {
                f.instruction(&Instruction::F64Eq);
            }
            (CompareOp::Ne, ValType::F64) | (CompareOp::FNe, ValType::F64) => {
                f.instruction(&Instruction::F64Ne);
            }
            (CompareOp::Lt, ValType::F64) | (CompareOp::FLt, ValType::F64) => {
                f.instruction(&Instruction::F64Lt);
            }
            (CompareOp::Le, ValType::F64) | (CompareOp::FLe, ValType::F64) => {
                f.instruction(&Instruction::F64Le);
            }
            (CompareOp::Gt, ValType::F64) | (CompareOp::FGt, ValType::F64) => {
                f.instruction(&Instruction::F64Gt);
            }
            (CompareOp::Ge, ValType::F64) | (CompareOp::FGe, ValType::F64) => {
                f.instruction(&Instruction::F64Ge);
            }
            (CompareOp::Eq, ValType::F32) | (CompareOp::FEq, ValType::F32) => {
                f.instruction(&Instruction::F32Eq);
            }
            (CompareOp::Ne, ValType::F32) | (CompareOp::FNe, ValType::F32) => {
                f.instruction(&Instruction::F32Ne);
            }
            (CompareOp::Lt, ValType::F32) | (CompareOp::FLt, ValType::F32) => {
                f.instruction(&Instruction::F32Lt);
            }
            (CompareOp::Le, ValType::F32) | (CompareOp::FLe, ValType::F32) => {
                f.instruction(&Instruction::F32Le);
            }
            (CompareOp::Gt, ValType::F32) | (CompareOp::FGt, ValType::F32) => {
                f.instruction(&Instruction::F32Gt);
            }
            (CompareOp::Ge, ValType::F32) | (CompareOp::FGe, ValType::F32) => {
                f.instruction(&Instruction::F32Ge);
            }

            (CompareOp::Eq, ValType::I32) => {
                f.instruction(&Instruction::I32Eq);
            }
            (CompareOp::Ne, ValType::I32) => {
                f.instruction(&Instruction::I32Ne);
            }
            (CompareOp::Lt, ValType::I32) => {
                f.instruction(&Instruction::I32LtS);
            }
            (CompareOp::Le, ValType::I32) => {
                f.instruction(&Instruction::I32LeS);
            }
            (CompareOp::Gt, ValType::I32) => {
                f.instruction(&Instruction::I32GtS);
            }
            (CompareOp::Ge, ValType::I32) => {
                f.instruction(&Instruction::I32GeS);
            }
            (CompareOp::ULt, ValType::I32) => {
                f.instruction(&Instruction::I32LtU);
            }
            (CompareOp::ULe, ValType::I32) => {
                f.instruction(&Instruction::I32LeU);
            }
            (CompareOp::UGt, ValType::I32) => {
                f.instruction(&Instruction::I32GtU);
            }
            (CompareOp::UGe, ValType::I32) => {
                f.instruction(&Instruction::I32GeU);
            }

            (CompareOp::Eq, ValType::I64) => {
                f.instruction(&Instruction::I64Eq);
            }
            (CompareOp::Ne, ValType::I64) => {
                f.instruction(&Instruction::I64Ne);
            }
            (CompareOp::Lt, ValType::I64) => {
                f.instruction(&Instruction::I64LtS);
            }
            (CompareOp::Le, ValType::I64) => {
                f.instruction(&Instruction::I64LeS);
            }
            (CompareOp::Gt, ValType::I64) => {
                f.instruction(&Instruction::I64GtS);
            }
            (CompareOp::Ge, ValType::I64) => {
                f.instruction(&Instruction::I64GeS);
            }
            (CompareOp::ULt, ValType::I64) => {
                f.instruction(&Instruction::I64LtU);
            }
            (CompareOp::ULe, ValType::I64) => {
                f.instruction(&Instruction::I64LeU);
            }
            (CompareOp::UGt, ValType::I64) => {
                f.instruction(&Instruction::I64GtU);
            }
            (CompareOp::UGe, ValType::I64) => {
                f.instruction(&Instruction::I64GeU);
            }

            (CompareOp::FEq, ValType::F32) => {
                f.instruction(&Instruction::F32Eq);
            }
            (CompareOp::FNe, ValType::F32) => {
                f.instruction(&Instruction::F32Ne);
            }
            (CompareOp::FLt, ValType::F32) => {
                f.instruction(&Instruction::F32Lt);
            }
            (CompareOp::FLe, ValType::F32) => {
                f.instruction(&Instruction::F32Le);
            }
            (CompareOp::FGt, ValType::F32) => {
                f.instruction(&Instruction::F32Gt);
            }
            (CompareOp::FGe, ValType::F32) => {
                f.instruction(&Instruction::F32Ge);
            }

            (CompareOp::FEq, ValType::F64) => {
                f.instruction(&Instruction::F64Eq);
            }
            (CompareOp::FNe, ValType::F64) => {
                f.instruction(&Instruction::F64Ne);
            }
            (CompareOp::FLt, ValType::F64) => {
                f.instruction(&Instruction::F64Lt);
            }
            (CompareOp::FLe, ValType::F64) => {
                f.instruction(&Instruction::F64Le);
            }
            (CompareOp::FGt, ValType::F64) => {
                f.instruction(&Instruction::F64Gt);
            }
            (CompareOp::FGe, ValType::F64) => {
                f.instruction(&Instruction::F64Ge);
            }

            // Ordered / unordered: Phase 1 approximation.
            (CompareOp::FOrd, _) => {
                f.instruction(&Instruction::I32Const(1));
                return;
            }
            (CompareOp::FUno, _) => {
                f.instruction(&Instruction::I32Const(0));
                return;
            }

            // Fallback: default to i32 comparisons.
            (CompareOp::Eq, _) => {
                f.instruction(&Instruction::I32Eq);
            }
            (CompareOp::Ne, _) => {
                f.instruction(&Instruction::I32Ne);
            }
            (CompareOp::Lt, _) => {
                f.instruction(&Instruction::I32LtS);
            }
            (CompareOp::Le, _) => {
                f.instruction(&Instruction::I32LeS);
            }
            (CompareOp::Gt, _) => {
                f.instruction(&Instruction::I32GtS);
            }
            (CompareOp::Ge, _) => {
                f.instruction(&Instruction::I32GeS);
            }
            (CompareOp::ULt, _) => {
                f.instruction(&Instruction::I32LtU);
            }
            (CompareOp::ULe, _) => {
                f.instruction(&Instruction::I32LeU);
            }
            (CompareOp::UGt, _) => {
                f.instruction(&Instruction::I32GtU);
            }
            (CompareOp::UGe, _) => {
                f.instruction(&Instruction::I32GeU);
            }
            (CompareOp::FEq, _) => {
                f.instruction(&Instruction::F64Eq);
            }
            (CompareOp::FNe, _) => {
                f.instruction(&Instruction::F64Ne);
            }
            (CompareOp::FLt, _) => {
                f.instruction(&Instruction::F64Lt);
            }
            (CompareOp::FLe, _) => {
                f.instruction(&Instruction::F64Le);
            }
            (CompareOp::FGt, _) => {
                f.instruction(&Instruction::F64Gt);
            }
            (CompareOp::FGe, _) => {
                f.instruction(&Instruction::F64Ge);
            }
        }
    }

    // ------------------------------------------------------------------
    // Cast emission
    // ------------------------------------------------------------------

    fn emit_cast(&self, f: &mut Function, from_ty: &IrType, to_ty: &IrType) {
        let from_vt = ir_type_to_wasm(from_ty);
        let to_vt = ir_type_to_wasm(to_ty);
        if from_vt == to_vt {
            return;
        }
        match (from_vt, to_vt) {
            (ValType::I32, ValType::I64) => {
                if from_ty.is_signed_integer() || matches!(from_ty, IrType::Bool) {
                    f.instruction(&Instruction::I64ExtendI32S);
                } else {
                    f.instruction(&Instruction::I64ExtendI32U);
                }
            }
            (ValType::I64, ValType::I32) => {
                f.instruction(&Instruction::I32WrapI64);
            }
            (ValType::I32, ValType::F32) => {
                f.instruction(&Instruction::F32ConvertI32S);
            }
            (ValType::I32, ValType::F64) => {
                f.instruction(&Instruction::F64ConvertI32S);
            }
            (ValType::I64, ValType::F64) => {
                f.instruction(&Instruction::F64ConvertI64S);
            }
            (ValType::I64, ValType::F32) => {
                f.instruction(&Instruction::F32ConvertI64S);
            }
            (ValType::F32, ValType::I32) => {
                f.instruction(&Instruction::I32TruncF32S);
            }
            (ValType::F64, ValType::I32) => {
                f.instruction(&Instruction::I32TruncF64S);
            }
            (ValType::F64, ValType::I64) => {
                f.instruction(&Instruction::I64TruncF64S);
            }
            (ValType::F32, ValType::I64) => {
                f.instruction(&Instruction::I64TruncF32S);
            }
            (ValType::F32, ValType::F64) => {
                f.instruction(&Instruction::F64PromoteF32);
            }
            (ValType::F64, ValType::F32) => {
                f.instruction(&Instruction::F32DemoteF64);
            }
            _ => {} // no-op
        }
    }

    // ------------------------------------------------------------------
    // Register get/set helpers
    // ------------------------------------------------------------------

    /// Emit a type conversion instruction if `from` != `to`.
    fn emit_type_coerce(&self, f: &mut Function, from: ValType, to: ValType) {
        if from == to {
            return;
        }
        match (from, to) {
            (ValType::I32, ValType::F64) => {
                f.instruction(&Instruction::F64ConvertI32S);
            }
            (ValType::F64, ValType::I32) => {
                f.instruction(&Instruction::I32TruncF64S);
            }
            (ValType::I32, ValType::F32) => {
                f.instruction(&Instruction::F32ConvertI32S);
            }
            (ValType::F32, ValType::I32) => {
                f.instruction(&Instruction::I32TruncF32S);
            }
            (ValType::F32, ValType::F64) => {
                f.instruction(&Instruction::F64PromoteF32);
            }
            (ValType::F64, ValType::F32) => {
                f.instruction(&Instruction::F32DemoteF64);
            }
            // Integer width conversions
            (ValType::I64, ValType::I32) => {
                f.instruction(&Instruction::I32WrapI64);
            }
            (ValType::I32, ValType::I64) => {
                f.instruction(&Instruction::I64ExtendI32S);
            }
            // Float↔I64 conversions
            (ValType::I64, ValType::F64) => {
                f.instruction(&Instruction::F64ConvertI64S);
            }
            (ValType::F64, ValType::I64) => {
                f.instruction(&Instruction::I64TruncF64S);
            }
            _ => {}
        }
    }

    fn get_reg(&self, f: &mut Function, id: IrId) {
        let idx = self.reg_to_local.get(&id).copied().unwrap_or(0);
        f.instruction(&Instruction::LocalGet(idx));
    }

    fn set_reg(&self, f: &mut Function, id: IrId) {
        let idx = self.reg_to_local.get(&id).copied().unwrap_or(0);
        f.instruction(&Instruction::LocalSet(idx));
    }
}

// ===========================================================================
// Free functions
// ===========================================================================

/// Map an IR type to a WASM value type.
/// Map an IR type to a WASM value type.
///
/// WASM32 mode: ALL integers (including I64/U64) map to i32.
/// Rayzor's `Int` is i64 in native but i32 in WASM32 — same as how
/// Haxe targets JavaScript with 32-bit integers. Only F64 stays 64-bit.
/// This eliminates the pointer width mismatch entirely.
fn ir_type_to_wasm(ty: &IrType) -> ValType {
    match ty {
        IrType::F32 => ValType::F32,
        IrType::F64 => ValType::F64,
        // SIMD vectors map to WASM v128. Currently only f32x4 (16 bytes) is
        // exercised by SIMD4f, but the same lowering applies to any 16-byte
        // vector type (i32x4, i16x8, i8x16, f64x2).
        IrType::Vector { .. } => ValType::V128,
        // Everything else is i32 in WASM32: integers, pointers, booleans,
        // strings, arrays, structs, unions, opaques, function refs.
        _ => ValType::I32,
    }
}

/// Allocation size for shadow-stack bumping.
fn wasm_alloc_size(ty: &IrType) -> i32 {
    match ty {
        IrType::Void => 0,
        IrType::Bool | IrType::I8 | IrType::U8 => 1,
        IrType::I16 | IrType::U16 => 2,
        IrType::I32 | IrType::U32 | IrType::F32 => 4,
        IrType::I64 | IrType::U64 | IrType::F64 => 8,
        IrType::Ptr(_) | IrType::Ref(_) => 4,
        IrType::Struct { fields, .. } => (fields.len() as i32) * 8,
        _ => 8,
    }
}

/// Byte stride for a GEP / PtrAdd index, derived from the pointee type.
///
/// On wasm the backend lays packed numeric buffers out at their natural element
/// width (f32/i32/u32 = 4, i16/u16 = 2, i8/u8 = 1), while pointer-sized slots —
/// `Ptr<I64>`/`Ptr<Void>` headers, array element slots, object fields — keep the
/// 8-byte stride the rest of the backend assumes. The previous inline rule
/// scaled *everything* non-byte by 8, so f32 tensor-buffer indexing strode 8
/// bytes per element and read every other lane as garbage. `ty` may be either a
/// `Ptr(inner)` (the common GEP shape) or a bare element type, so handle both.
fn gep_pointee_size(ty: &IrType) -> i32 {
    fn size_of(t: &IrType) -> i32 {
        match t {
            IrType::U8 | IrType::I8 => 1,
            IrType::U16 | IrType::I16 => 2,
            IrType::U32 | IrType::I32 | IrType::F32 => 4,
            // I64/U64/F64/Bool/Ptr/Void/struct/etc. — 8-byte slot layout.
            _ => 8,
        }
    }
    match ty {
        IrType::Ptr(inner) => size_of(inner),
        other => size_of(other),
    }
}

/// Convert an IrValue to a raw i64 (for global initializers).
fn ir_value_to_i64(val: &IrValue) -> i64 {
    match val {
        IrValue::Void | IrValue::Undef | IrValue::Null => 0,
        IrValue::Bool(b) => *b as i64,
        IrValue::I8(v) => *v as i64,
        IrValue::I16(v) => *v as i64,
        IrValue::I32(v) => *v as i64,
        IrValue::I64(v) => *v,
        IrValue::U8(v) => *v as i64,
        IrValue::U16(v) => *v as i64,
        IrValue::U32(v) => *v as i64,
        IrValue::U64(v) => *v as i64,
        IrValue::F32(v) => (*v as f64).to_bits() as i64,
        IrValue::F64(v) => v.to_bits() as i64,
        _ => 0,
    }
}

/// Emit a zero constant for the given WASM type.
fn emit_zero(f: &mut Function, vt: ValType) {
    match vt {
        ValType::I32 => {
            f.instruction(&Instruction::I32Const(0));
        }
        ValType::I64 => {
            f.instruction(&Instruction::I64Const(0));
        }
        ValType::F32 => {
            f.instruction(&Instruction::F32Const(Ieee32::from(0.0f32)));
        }
        ValType::F64 => {
            f.instruction(&Instruction::F64Const(Ieee64::from(0.0f64)));
        }
        _ => {
            f.instruction(&Instruction::I32Const(0));
        }
    }
}

/// Is this IR vector type the f32x4 shape that RelaxedMadd targets?
/// Currently only f32x4 is wired through wasm_backend SIMD lowering
/// (count == 4 elements, element type F32). The fused-multiply-add
/// peephole gates on this so we don't try to emit RelaxedMadd against
/// a hypothetical f64x2 or integer SIMD shape.
fn is_f32x4_vec(ty: &IrType) -> bool {
    matches!(ty, IrType::Vector { element, count } if **element == IrType::F32 && *count == 4)
}

/// Emit a bitcast between two WASM types.
fn emit_bitcast(f: &mut Function, from: ValType, to: ValType) {
    if from == to {
        return;
    }
    match (from, to) {
        (ValType::I32, ValType::F32) => {
            f.instruction(&Instruction::F32ReinterpretI32);
        }
        (ValType::F32, ValType::I32) => {
            f.instruction(&Instruction::I32ReinterpretF32);
        }
        (ValType::I64, ValType::F64) => {
            f.instruction(&Instruction::F64ReinterpretI64);
        }
        (ValType::F64, ValType::I64) => {
            f.instruction(&Instruction::I64ReinterpretF64);
        }
        (ValType::I32, ValType::I64) => {
            f.instruction(&Instruction::I64ExtendI32U);
        }
        (ValType::I64, ValType::I32) => {
            f.instruction(&Instruction::I32WrapI64);
        }
        // Cross-width bitcasts (through I64 intermediate)
        (ValType::F64, ValType::I32) => {
            f.instruction(&Instruction::I64ReinterpretF64);
            f.instruction(&Instruction::I32WrapI64);
        }
        (ValType::I32, ValType::F64) => {
            f.instruction(&Instruction::I64ExtendI32U);
            f.instruction(&Instruction::F64ReinterpretI64);
        }
        _ => {} // no-op best effort
    }
}

/// Emit a typed load at offset 0.
fn emit_typed_load(f: &mut Function, vt: ValType) {
    let ma = MemArg {
        offset: 0,
        align: 0,
        memory_index: 0,
    };
    match vt {
        ValType::I64 => {
            f.instruction(&Instruction::I64Load(ma));
        }
        ValType::F32 => {
            f.instruction(&Instruction::F32Load(ma));
        }
        ValType::F64 => {
            f.instruction(&Instruction::F64Load(ma));
        }
        _ => {
            f.instruction(&Instruction::I32Load(ma));
        }
    }
}

/// Emit a typed load at a constant offset.
fn emit_typed_load_offset(f: &mut Function, vt: ValType, offset: u64) {
    let ma = MemArg {
        offset,
        align: 0,
        memory_index: 0,
    };
    match vt {
        ValType::I64 => {
            f.instruction(&Instruction::I64Load(ma));
        }
        ValType::F32 => {
            f.instruction(&Instruction::F32Load(ma));
        }
        ValType::F64 => {
            f.instruction(&Instruction::F64Load(ma));
        }
        _ => {
            f.instruction(&Instruction::I32Load(ma));
        }
    }
}

/// Emit a typed store at offset 0.
fn emit_typed_store(f: &mut Function, vt: ValType) {
    let ma = MemArg {
        offset: 0,
        align: 0,
        memory_index: 0,
    };
    match vt {
        ValType::I64 => {
            f.instruction(&Instruction::I64Store(ma));
        }
        ValType::F32 => {
            f.instruction(&Instruction::F32Store(ma));
        }
        ValType::F64 => {
            f.instruction(&Instruction::F64Store(ma));
        }
        _ => {
            f.instruction(&Instruction::I32Store(ma));
        }
    }
}

/// Emit a typed store at a constant offset.
fn emit_typed_store_offset(f: &mut Function, vt: ValType, offset: u64) {
    let ma = MemArg {
        offset,
        align: 0,
        memory_index: 0,
    };
    match vt {
        ValType::I64 => {
            f.instruction(&Instruction::I64Store(ma));
        }
        ValType::F32 => {
            f.instruction(&Instruction::F32Store(ma));
        }
        ValType::F64 => {
            f.instruction(&Instruction::F64Store(ma));
        }
        _ => {
            f.instruction(&Instruction::I32Store(ma));
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_type_to_wasm() {
        assert_eq!(ir_type_to_wasm(&IrType::I32), ValType::I32);
        // In WASM32, I64 maps to I32 (all pointers and type-erased values are 32-bit)
        assert_eq!(ir_type_to_wasm(&IrType::I64), ValType::I32);
        assert_eq!(ir_type_to_wasm(&IrType::F32), ValType::F32);
        assert_eq!(ir_type_to_wasm(&IrType::F64), ValType::F64);
        assert_eq!(ir_type_to_wasm(&IrType::Bool), ValType::I32);
        assert_eq!(
            ir_type_to_wasm(&IrType::Ptr(Box::new(IrType::I32))),
            ValType::I32
        );
        assert_eq!(ir_type_to_wasm(&IrType::String), ValType::I32);
        assert_eq!(ir_type_to_wasm(&IrType::Any), ValType::I32);
    }

    #[test]
    fn test_empty_module_compiles() {
        let module = IrModule::new("test".to_string(), "test.hx".to_string());
        let modules: Vec<&IrModule> = vec![&module];
        let result = WasmBackend::compile(&modules, None);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        // Valid WASM starts with magic: \0asm
        assert!(bytes.len() >= 8);
        assert_eq!(&bytes[0..4], b"\0asm");
    }
}
