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
use std::collections::{BTreeMap, BTreeSet, HashMap};

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, GlobalSection, GlobalType, Ieee32, Ieee64, ImportSection,
    Instruction, MemArg, MemorySection, MemoryType, Module, TypeSection, ValType,
};

use crate::ir::blocks::{IrBasicBlock, IrBlockId, IrTerminator};
use crate::ir::functions::{IrFunction, IrFunctionId, IrFunctionSignature};
use crate::ir::instructions::{BinaryOp, CompareOp, IrInstruction, UnaryOp};
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
        Self::compile_with_exports(modules, entry_function, &[])
    }

    /// Compile with additional function exports (for JS interop).
    /// `extra_exports` is a list of function names to export from WASM.
    pub fn compile_with_exports(
        modules: &[&IrModule],
        entry_function: Option<&str>,
        extra_exports: &[&str],
    ) -> Result<Vec<u8>, String> {
        let mut ctx = CompileCtx::new();
        ctx.collect_imports(modules);
        ctx.collect_functions(modules);
        ctx.collect_strings(modules);
        ctx.collect_globals(modules);
        // Build set of exported class names from @:export functions
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
    type_map: HashMap<(Vec<ValType>, Vec<ValType>), u32>,
    types: Vec<(Vec<ValType>, Vec<ValType>)>,

    // ----- Import section -----
    imports: Vec<ImportedFunc>,
    import_name_to_idx: HashMap<String, u32>,

    // ----- Internal functions -----
    internals: Vec<InternalFunc>,
    /// IrFunctionId -> absolute WASM function index.
    ir_func_to_idx: HashMap<IrFunctionId, u32>,
    /// Name -> absolute func_idx for entry-point lookup.
    func_name_to_idx: HashMap<String, u32>,

    next_func_idx: u32,

    // ----- String pool / data section -----
    string_entries: Vec<DataString>,
    string_offsets: HashMap<String, u32>,
    data_offset: u32,

    // ----- Globals -----
    ir_global_to_idx: HashMap<IrGlobalId, u32>,
    /// User globals (excluding __stack_pointer which is always index 0).
    user_globals: Vec<(IrGlobalId, ValType, i64)>,

    // ----- Function return types -----
    /// IrFunctionId -> WASM return type. Used for CallDirect dest type inference.
    func_return_types: HashMap<IrFunctionId, ValType>,
    /// IrFunctionId -> WASM parameter types. Used for CallDirect arg coercion.
    func_param_types: HashMap<IrFunctionId, Vec<ValType>>,
    /// Class names that have @:export (for constructor detection).
    exported_classes: std::collections::HashSet<String>,
}

impl CompileCtx {
    fn new() -> Self {
        Self {
            type_map: HashMap::new(),
            types: Vec::new(),
            imports: Vec::new(),
            import_name_to_idx: HashMap::new(),
            internals: Vec::new(),
            ir_func_to_idx: HashMap::new(),
            func_name_to_idx: HashMap::new(),
            next_func_idx: 0,
            string_entries: Vec::new(),
            string_offsets: HashMap::new(),
            data_offset: DATA_SECTION_BASE,
            ir_global_to_idx: HashMap::new(),
            user_globals: Vec::new(),
            func_return_types: HashMap::new(),
            func_param_types: HashMap::new(),
            exported_classes: std::collections::HashSet::new(),
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

    // ------------------------------------------------------------------
    // Phase 1a -- imports (extern functions that have NO internal body)
    // ------------------------------------------------------------------

    fn collect_imports(&mut self, modules: &[&IrModule]) {
        // First, build a set of IrFunctionIds that have actual code bodies.
        // These are internal functions and should NOT be imported.
        let mut has_code: BTreeSet<IrFunctionId> = BTreeSet::new();
        let mut has_code_name: BTreeSet<String> = BTreeSet::new();
        for module in modules {
            for (func_id, func) in &module.functions {
                if !func.cfg.blocks.is_empty() {
                    has_code.insert(*func_id);
                    has_code_name.insert(func.name.clone());
                }
            }
        }

        let mut seen: BTreeSet<String> = BTreeSet::new();

        // Import declared externs that have no internal body.
        for module in modules {
            for (_id, ext) in &module.extern_functions {
                if has_code.contains(&ext.id) || has_code_name.contains(&ext.name) {
                    continue;
                }
                if seen.contains(&ext.name) {
                    if let Some(&idx) = self.import_name_to_idx.get(&ext.name) {
                        self.ir_func_to_idx.insert(ext.id, idx);
                    }
                    continue;
                }
                seen.insert(ext.name.clone());
                let (params, results) = Self::sig_to_wasm(&ext.signature);
                let type_idx = self.intern_type(params, results);
                let func_idx = self.next_func_idx;
                self.next_func_idx += 1;
                self.imports.push(ImportedFunc {
                    name: ext.name.clone(),
                    type_idx,
                });
                self.import_name_to_idx.insert(ext.name.clone(), func_idx);
                self.ir_func_to_idx.insert(ext.id, func_idx);
            }
        }

        // Also import empty-body functions from module.functions — these are
        // MIR wrappers for runtime functions (like haxe_trace_string_struct).
        // In native mode, the cranelift backend links them to runtime symbols.
        // In WASM mode, they become host imports.
        for module in modules {
            for (func_id, func) in &module.functions {
                if func.cfg.blocks.is_empty()
                    && !self.ir_func_to_idx.contains_key(func_id)
                    && !seen.contains(&func.name)
                {
                    seen.insert(func.name.clone());
                    let (params, results) = Self::sig_to_wasm(&func.signature);
                    let type_idx = self.intern_type(params, results);
                    let func_idx = self.next_func_idx;
                    self.next_func_idx += 1;
                    self.imports.push(ImportedFunc {
                        name: func.name.clone(),
                        type_idx,
                    });
                    self.import_name_to_idx.insert(func.name.clone(), func_idx);
                    self.ir_func_to_idx.insert(*func_id, func_idx);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 1b -- internal functions
    // ------------------------------------------------------------------

    fn collect_functions(&mut self, modules: &[&IrModule]) {
        for (mod_idx, module) in modules.iter().enumerate() {
            for (func_id, func) in &module.functions {
                // Skip functions already registered as imports (by ID or name).
                // Always register return type for CallDirect type inference
                let ret_vt = ir_type_to_wasm(&func.signature.return_type);
                self.func_return_types.insert(*func_id, ret_vt);
                let param_vts: Vec<ValType> = func
                    .signature
                    .parameters
                    .iter()
                    .map(|p| ir_type_to_wasm(&p.ty))
                    .collect();
                self.func_param_types.insert(*func_id, param_vts);

                if self.ir_func_to_idx.contains_key(func_id) {
                    let idx = *self.ir_func_to_idx.get(func_id).unwrap();
                    self.func_name_to_idx.insert(func.name.clone(), idx);
                    continue;
                }
                if let Some(&idx) = self.import_name_to_idx.get(&func.name) {
                    self.ir_func_to_idx.insert(*func_id, idx);
                    self.func_name_to_idx.insert(func.name.clone(), idx);
                    continue;
                }
                let (params, results) = Self::sig_to_wasm(&func.signature);
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
        let mut import_section = ImportSection::new();
        for imp in &self.imports {
            import_section.import("rayzor", &imp.name, EntityType::Function(imp.type_idx));
        }
        wasm_module.section(&import_section);

        // --- Function section ---
        let mut func_section = FunctionSection::new();
        for internal in &self.internals {
            func_section.function(internal.type_idx);
        }
        wasm_module.section(&func_section);

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
        let mut export_section = ExportSection::new();
        export_section.export("memory", ExportKind::Memory, 0);
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
                export_section.export("_start", ExportKind::Func, idx);
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
            let body = FunctionLowerer::lower(self, ir_func)?;
            code_section.function(&body);
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
    reg_to_local: HashMap<IrId, u32>,

    /// Locals to declare (count, type) groups. Built during allocation.
    local_types: Vec<ValType>,

    /// Number of WASM parameters (locals 0..param_count-1).
    param_count: u32,

    /// Next available local index.
    next_local: u32,

    /// Ordered block IDs (entry first, then sorted by id).
    block_order: Vec<IrBlockId>,
    /// Block ID -> positional index in block_order.
    block_index: HashMap<IrBlockId, u32>,
}

impl<'a> FunctionLowerer<'a> {
    fn lower(ctx: &'a CompileCtx, ir_func: &'a IrFunction) -> Result<Function, String> {
        let param_count = ir_func.signature.parameters.len() as u32;
        let mut low = Self {
            ctx,
            ir_func,
            reg_to_local: HashMap::new(),
            local_types: Vec::new(),
            param_count,
            next_local: param_count,
            block_order: Vec::new(),
            block_index: HashMap::new(),
        };

        // Early return for empty-body functions (extern stubs).
        if ir_func.cfg.blocks.is_empty() {
            return low.build_body();
        }

        // Map parameter registers.
        for (i, param) in ir_func.signature.parameters.iter().enumerate() {
            low.reg_to_local.insert(param.reg, i as u32);
        }

        // Pre-allocate locals for every register used in the function.
        low.allocate_locals();

        // Compute block order.
        low.compute_block_order();

        // Build the function body.
        low.build_body()
    }

    // ------------------------------------------------------------------
    // Local allocation
    // ------------------------------------------------------------------

    fn allocate_locals(&mut self) {
        // Step 1: Allocate all locals as I32 (default WASM32 type).
        let mut seen = BTreeSet::new();
        for p in &self.ir_func.signature.parameters {
            seen.insert(p.reg);
        }
        for block in self.ir_func.cfg.blocks.values() {
            for phi in &block.phi_nodes {
                if seen.insert(phi.dest) {
                    self.alloc_local(phi.dest, ValType::I32);
                }
            }
            for inst in &block.instructions {
                if let Some(dest) = inst.dest() {
                    if seen.insert(dest) {
                        self.alloc_local(dest, ValType::I32);
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

                        // CallDirect: use callee return type
                        IrInstruction::CallDirect { func_id, .. } => {
                            self.ctx.func_return_types.get(func_id).copied()
                        }

                        // BinOp: float if any operand is float
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
                            if is_fop || lt == ValType::F64 || rt == ValType::F64 {
                                Some(ValType::F64)
                            } else if lt == ValType::F32 || rt == ValType::F32 {
                                Some(ValType::F32)
                            } else {
                                None // keep I32
                            }
                        }

                        // UnOp: inherit operand type
                        IrInstruction::UnOp { op, operand, .. } => {
                            let is_fop = matches!(op, UnaryOp::FNeg);
                            let ot = self.local_type_of(*operand).unwrap_or(ValType::I32);
                            if is_fop || ot == ValType::F64 {
                                Some(ValType::F64)
                            } else if ot == ValType::F32 {
                                Some(ValType::F32)
                            } else {
                                None
                            }
                        }

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
    }

    /// Get the WASM type of an IrId's allocated local.
    fn local_type_of(&self, id: IrId) -> Option<ValType> {
        let &local_idx = self.reg_to_local.get(&id)?;
        if local_idx < self.param_count {
            let param_idx = local_idx as usize;
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

    fn build_single_block(&mut self) -> Result<Function, String> {
        let bid = self.block_order[0];
        let block = self.ir_func.cfg.blocks.get(&bid).unwrap();

        let mut f = Function::new(self.local_types.iter().map(|t| (1, *t)));

        for inst in &block.instructions {
            self.emit_instruction(&mut f, inst);
        }

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

            // Emit instructions.
            for inst in &block.instructions {
                self.emit_instruction(&mut f, inst);
            }

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
                }
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
                }
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
                self.get_reg(f, *src);
                self.emit_type_coerce(f, self.reg_wasm_type(*src), self.reg_wasm_type(*dest));
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

                // Determine the operation type
                let is_fop = matches!(
                    op,
                    BinaryOp::FAdd
                        | BinaryOp::FSub
                        | BinaryOp::FMul
                        | BinaryOp::FDiv
                        | BinaryOp::FRem
                );
                let op_ty = if is_fop
                    || dest_ty == ValType::F64
                    || left_ty == ValType::F64
                    || right_ty == ValType::F64
                {
                    ValType::F64
                } else if dest_ty == ValType::F32
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
            IrInstruction::BitCast { dest, src, .. } => {
                self.get_reg(f, *src);
                let from_vt = self.reg_wasm_type(*src);
                let to_vt = self.reg_wasm_type(*dest);
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
                let vt = ir_type_to_wasm(actual_ty);
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
                let elem_size: i32 = match ty {
                    IrType::Ptr(inner) => match inner.as_ref() {
                        IrType::U8 | IrType::I8 => 1,
                        _ => 8,
                    },
                    IrType::U8 | IrType::I8 => 1,
                    _ => 8,
                };
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
                dest, ptr, offset, ..
            } => {
                self.get_reg(f, *ptr);
                self.get_reg(f, *offset);
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

            // === Free (no-op in Phase 1) ===
            IrInstruction::Free { .. } => {}

            // === CallDirect ===
            IrInstruction::CallDirect {
                dest,
                func_id,
                args,
                ..
            } => {
                // Load args with type coercion to match callee signature
                let callee_params = self.ctx.func_param_types.get(func_id);
                for (i, arg) in args.iter().enumerate() {
                    self.get_reg(f, *arg);
                    let arg_ty = self.reg_wasm_type(*arg);
                    if let Some(params) = callee_params {
                        if let Some(&expected) = params.get(i) as Option<&ValType> {
                            self.emit_type_coerce(f, arg_ty, expected);
                        }
                    }
                }
                if let Some(&idx) = self.ctx.ir_func_to_idx.get(func_id) {
                    f.instruction(&Instruction::Call(idx));
                } else {
                    // Unknown function -- drop args, trap.
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
                for arg in args {
                    self.get_reg(f, *arg);
                }
                self.get_reg(f, *func_ptr);
                let (params, results) = match signature {
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
                    _ => {
                        let p: Vec<ValType> = args.iter().map(|a| self.reg_wasm_type(*a)).collect();
                        let r = dest
                            .map(|d| vec![self.reg_wasm_type(d)])
                            .unwrap_or_default();
                        (p, r)
                    }
                };
                let type_idx = self
                    .ctx
                    .type_map
                    .get(&(params, results))
                    .copied()
                    .unwrap_or(0);
                f.instruction(&Instruction::CallIndirect {
                    type_index: type_idx,
                    table_index: 0,
                });
                if let Some(d) = dest {
                    self.set_reg(f, *d);
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
                f.instruction(&Instruction::I32Const(idx as i32));
                self.set_reg(f, *dest);
            }

            // === MakeClosure ===
            IrInstruction::MakeClosure {
                dest,
                func_id,
                captured_values,
            } => {
                // Layout: [fn_idx: i32] [padding: i32] [captures...]
                let slot_count = 2 + captured_values.len();
                let alloc_size = ((slot_count * 4 + 7) & !7) as i32;

                // Bump shadow stack.
                f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                f.instruction(&Instruction::I32Const(alloc_size));
                f.instruction(&Instruction::I32Sub);
                f.instruction(&Instruction::GlobalSet(STACK_PTR_GLOBAL));

                // Store fn_idx at offset 0.
                f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                let fn_idx = self.ctx.ir_func_to_idx.get(func_id).copied().unwrap_or(0);
                f.instruction(&Instruction::I32Const(fn_idx as i32));
                f.instruction(&Instruction::I32Store(MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }));

                // Store captured values at offsets 8, 12, 16, ...
                for (i, cap) in captured_values.iter().enumerate() {
                    f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                    self.get_reg(f, *cap);
                    f.instruction(&Instruction::I32Store(MemArg {
                        offset: (8 + i * 4) as u64,
                        align: 2,
                        memory_index: 0,
                    }));
                }

                f.instruction(&Instruction::GlobalGet(STACK_PTR_GLOBAL));
                self.set_reg(f, *dest);
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
                // Env starts at offset 8 in our closure layout.
                self.get_reg(f, *closure);
                f.instruction(&Instruction::I32Const(8));
                f.instruction(&Instruction::I32Add);
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
                }
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

            // === InlineAsm / SIMD (unsupported in Phase 1) ===
            IrInstruction::InlineAsm { .. }
            | IrInstruction::VectorLoad { .. }
            | IrInstruction::VectorStore { .. }
            | IrInstruction::VectorBinOp { .. }
            | IrInstruction::VectorSplat { .. }
            | IrInstruction::VectorExtract { .. }
            | IrInstruction::VectorInsert { .. }
            | IrInstruction::VectorReduce { .. }
            | IrInstruction::VectorUnaryOp { .. }
            | IrInstruction::VectorMinMax { .. } => {
                f.instruction(&Instruction::Unreachable);
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
        assert_eq!(ir_type_to_wasm(&IrType::I64), ValType::I64);
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
