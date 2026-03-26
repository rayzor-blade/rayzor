//! WebAssembly backend — compiles MIR to WASM binary via wasm-encoder.
//!
//! Produces standalone `.wasm` modules that run in browsers (via JS glue)
//! and server-side (via wasmtime/WASI). Uses the WASM Component Model
//! for rpkg package distribution.
//!
//! # Architecture
//!
//! MIR (IrModule) → WasmBackend::compile() → Vec<u8> (.wasm binary)
//!
//! The backend handles:
//! - MIR instruction → WASM instruction mapping
//! - CFG → structured control flow (block/loop/br) via Relooper
//! - Phi nodes → explicit WASM locals
//! - Runtime imports for stdlib functions
//! - Linear memory management (stack + heap)

use crate::ir::{IrFunctionId, IrModule};
use std::collections::BTreeMap;
use wasm_encoder::{
    CodeSection, DataSection, ExportKind, ExportSection, Function, FunctionSection, GlobalSection,
    GlobalType, ImportSection, Instruction, MemorySection, MemoryType, Module, TypeSection, ValType,
};

/// WebAssembly compilation backend.
pub struct WasmBackend {
    /// MIR function ID → WASM function index
    function_map: BTreeMap<IrFunctionId, u32>,
    /// Number of imported functions (offsets user function indices)
    import_count: u32,
}

impl WasmBackend {
    /// Compile MIR modules to a WASM binary.
    ///
    /// Returns the raw `.wasm` bytes ready to write to disk or execute via wasmtime.
    pub fn compile(
        modules: &[&IrModule],
        entry_function: Option<&str>,
    ) -> Result<Vec<u8>, String> {
        let mut backend = Self {
            function_map: BTreeMap::new(),
            import_count: 0,
        };

        let mut wasm_module = Module::new();

        // --- Type section ---
        let mut types = TypeSection::new();
        // Type 0: () -> () (for _start / main)
        types.ty().function(vec![], vec![]);
        // Type 1: (i32) -> () (for trace)
        types.ty().function(vec![ValType::I32], vec![]);
        // Type 2: (i32) -> i32 (for alloc)
        types.ty().function(vec![ValType::I32], vec![ValType::I32]);
        wasm_module.section(&types);

        // --- Import section ---
        let mut imports = ImportSection::new();
        // Import trace function from host
        imports.import("rayzor", "trace", wasm_encoder::EntityType::Function(1));
        backend.import_count = 1;
        wasm_module.section(&imports);

        // --- Function section ---
        let mut functions = FunctionSection::new();
        // Collect all functions from all modules
        let mut all_functions = Vec::new();
        let mut entry_func_idx = None;

        for module in modules {
            for (func_id, func) in &module.functions {
                let func_idx = backend.import_count + all_functions.len() as u32;
                backend.function_map.insert(*func_id, func_idx);

                // Check if this is the entry function
                if let Some(entry) = entry_function {
                    if func.name == entry
                        || func.name.ends_with("::main")
                        || func.qualified_name.as_deref() == Some(entry)
                    {
                        entry_func_idx = Some(func_idx);
                    }
                }

                functions.function(0); // All use type 0 for now (simplified)
                all_functions.push((*func_id, func));
            }
        }
        wasm_module.section(&functions);

        // --- Memory section ---
        let mut memories = MemorySection::new();
        memories.memory(MemoryType {
            minimum: 256,    // 16MB initial
            maximum: Some(65536), // 4GB max
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        wasm_module.section(&memories);

        // --- Global section ---
        let mut globals = GlobalSection::new();
        // __stack_pointer: starts at end of initial memory minus 4KB
        globals.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: true,
                shared: false,
            },
            &wasm_encoder::ConstExpr::i32_const((256 * 65536 - 4096) as i32),
        );
        wasm_module.section(&globals);

        // --- Export section ---
        let mut exports = ExportSection::new();
        exports.export("memory", ExportKind::Memory, 0);
        if let Some(idx) = entry_func_idx {
            exports.export("_start", ExportKind::Func, idx);
        }
        wasm_module.section(&exports);

        // --- Code section ---
        let mut codes = CodeSection::new();
        for (_func_id, _func) in &all_functions {
            let mut f = Function::new(vec![]);
            // Phase 1: emit a minimal function body
            // TODO: lower MIR instructions to WASM
            f.instruction(&Instruction::End);
            codes.function(&f);
        }
        wasm_module.section(&codes);

        // --- Data section (string pool) ---
        let data = DataSection::new();
        wasm_module.section(&data);

        Ok(wasm_module.finish())
    }
}

/// Convert an IrType to a WASM ValType.
fn _ir_type_to_wasm(ty: &crate::ir::IrType) -> ValType {
    use crate::ir::IrType;
    match ty {
        IrType::Bool | IrType::I8 | IrType::I16 | IrType::I32 | IrType::U8 | IrType::U16 | IrType::U32 => ValType::I32,
        IrType::I64 | IrType::U64 => ValType::I64,
        IrType::F32 => ValType::F32,
        IrType::F64 => ValType::F64,
        IrType::Ptr(_) | IrType::Ref(_) => ValType::I32, // WASM32 linear memory address
        IrType::Void => ValType::I32, // void returns handled separately
        _ => ValType::I32, // String, Array, Struct, Any → pointer (i32)
    }
}
