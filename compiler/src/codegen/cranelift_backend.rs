/// Cranelift JIT Backend
///
/// This backend translates MIR to Cranelift IR and performs JIT compilation.
/// Used for:
/// - Cold path execution (first call of functions)
/// - Development mode (fast iteration)
/// - Testing
///
/// Performance targets:
/// - Compilation: 50-200ms per function
/// - Runtime: 15-25x interpreter speed
use cranelift::prelude::*;
use cranelift_codegen::ir::{ArgumentPurpose, BlockArg, Function};
use cranelift_codegen::settings;
use cranelift_frontend::Variable;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_native;
use std::collections::{HashMap, HashSet};

use crate::ir::{
    IrBasicBlock, IrBlockId, IrControlFlowGraph, IrFunction, IrFunctionId, IrId, IrInstruction,
    IrModule, IrTerminator, IrType, IrValue,
};
use tracing::{debug, info, trace, warn};

/// Cranelift JIT backend for compiling MIR to native code
pub struct CraneliftBackend {
    /// Cranelift JIT module
    module: JITModule,

    /// Cranelift codegen context
    ctx: codegen::Context,

    /// Map from MIR function IDs to Cranelift function IDs
    function_map: HashMap<IrFunctionId, FuncId>,

    /// Map from MIR value IDs to Cranelift values (per function)
    pub(super) value_map: HashMap<IrId, Value>,

    /// Map from closure registers (function pointers) to their environment pointers
    /// This is populated during MakeClosure and used during CallIndirect
    closure_environments: HashMap<IrId, Value>,

    /// Map from runtime function names to Cranelift function IDs
    /// Used for rayzor_malloc, rayzor_realloc, rayzor_free
    runtime_functions: HashMap<String, FuncId>,

    /// Target pointer size (32-bit or 64-bit) from ISA
    pointer_type: types::Type,

    /// Module counter for unique function naming across multiple MIR modules
    /// Each MIR module starts function IDs from 0, so we need to disambiguate
    module_counter: usize,

    /// Set of Cranelift FuncIds that have already been defined (had their bodies compiled)
    /// Used to prevent duplicate definition errors when functions are shared across modules
    /// We track by FuncId (not name) because different modules can have functions with the
    /// same MIR name (e.g., 'new') but different Cranelift symbols (e.g., m1_func_0 vs m3_func_0)
    defined_functions: HashSet<FuncId>,

    /// The environment parameter for the current function being compiled
    /// This is used by ClosureEnv to access the environment
    current_env_param: Option<Value>,

    /// Map from string content to its DataId in the module
    /// Used to reuse string constants across functions
    string_data: HashMap<String, DataId>,

    /// Counter for unique string data names
    string_counter: usize,

    /// Map from qualified function names to Cranelift function IDs
    /// Used to link forward references to actual implementations
    /// Key is qualified name (e.g., "StringTools.unsafeCodeAt")
    qualified_name_to_func: HashMap<String, FuncId>,
}

impl CraneliftBackend {
    /// Create a new Cranelift backend with default optimization level (speed)
    pub fn new() -> Result<Self, String> {
        Self::with_symbols(&[])
    }

    /// Create a new Cranelift backend with custom runtime symbols from plugins
    pub fn with_symbols(symbols: &[(&str, *const u8)]) -> Result<Self, String> {
        Self::with_symbols_and_opt("speed", symbols)
    }

    /// Create a new Cranelift backend with specified optimization level
    ///
    /// Optimization levels:
    /// - "none": No optimization (Tier 0 - Baseline)
    /// - "speed": Moderate optimization (Tier 1 - Standard)
    /// - "speed_and_size": Aggressive optimization (Tier 2 - Optimized)
    pub fn with_optimization_level(opt_level: &str) -> Result<Self, String> {
        Self::with_symbols_and_opt(opt_level, &[])
    }

    /// Create a fast compilation backend for development (no optimization)
    ///
    /// This uses "none" optimization level for fastest compilation.
    /// Suitable for development iteration where compile time matters more than runtime speed.
    pub fn with_fast_compilation(symbols: &[(&str, *const u8)]) -> Result<Self, String> {
        Self::with_symbols_and_opt("none", symbols)
    }

    /// Create backend with symbols and optimization level
    pub fn with_symbols_and_opt(
        opt_level: &str,
        symbols: &[(&str, *const u8)],
    ) -> Result<Self, String> {
        // Configure Cranelift for the current platform
        let mut flag_builder = settings::builder();

        // Disable colocated libcalls for compatibility with ARM64 (Apple Silicon)
        // This prevents PLT usage which is only supported on x86_64
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| format!("Failed to set use_colocated_libcalls: {}", e))?;

        // Disable PIC (Position Independent Code) for simpler code generation
        // Non-PIC code is faster due to fewer indirections
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| format!("Failed to set is_pic: {}", e))?;

        // Set optimization level (configurable for tiered compilation)
        flag_builder
            .set("opt_level", opt_level)
            .map_err(|e| format!("Failed to set opt_level: {}", e))?;

        // Enable tier-specific optimizations
        // Note: SIMD vectorization is automatically enabled through cranelift_native::builder()
        // which detects CPU features (AVX2, NEON, etc.) at runtime
        match opt_level {
            "none" => {
                // Baseline tier: fastest compilation, minimal optimization
                // Use single-pass register allocation for faster compilation
                flag_builder
                    .set("regalloc_algorithm", "single_pass")
                    .map_err(|e| format!("Failed to set regalloc_algorithm: {}", e))?;

                // Verifier enabled in debug only for fast iteration
                #[cfg(debug_assertions)]
                flag_builder
                    .set("enable_verifier", "true")
                    .map_err(|e| format!("Failed to set enable_verifier: {}", e))?;
                #[cfg(not(debug_assertions))]
                flag_builder
                    .set("enable_verifier", "false")
                    .map_err(|e| format!("Failed to set enable_verifier: {}", e))?;
            }
            "speed" => {
                // Standard tier: moderate optimization
                // Disable verifier for faster compilation at higher tiers
                flag_builder
                    .set("enable_verifier", "false")
                    .map_err(|e| format!("Failed to set enable_verifier: {}", e))?;

                // Disable frame pointers for slightly smaller/faster code
                flag_builder
                    .set("preserve_frame_pointers", "false")
                    .map_err(|e| format!("Failed to set preserve_frame_pointers: {}", e))?;
            }
            "speed_and_size" => {
                // Optimized tier: aggressive optimization
                // All optimizations enabled for maximum performance
                flag_builder
                    .set("enable_verifier", "false")
                    .map_err(|e| format!("Failed to set enable_verifier: {}", e))?;

                // Disable frame pointers for smaller/faster code
                flag_builder
                    .set("preserve_frame_pointers", "false")
                    .map_err(|e| format!("Failed to set preserve_frame_pointers: {}", e))?;

                // Enable probestack for large stack allocations (prevents stack overflow)
                flag_builder
                    .set("enable_probestack", "true")
                    .map_err(|e| format!("Failed to set enable_probestack: {}", e))?;
            }
            _ => {
                // Unknown level, use safe defaults
                #[cfg(debug_assertions)]
                flag_builder
                    .set("enable_verifier", "true")
                    .map_err(|e| format!("Failed to set enable_verifier: {}", e))?;
                #[cfg(not(debug_assertions))]
                flag_builder
                    .set("enable_verifier", "false")
                    .map_err(|e| format!("Failed to set enable_verifier: {}", e))?;
            }
        }

        // Create ISA for the current platform with native feature detection
        // This automatically enables CPU-specific features (AVX2, NEON, etc.)
        let isa_builder = cranelift_native::builder()
            .map_err(|e| format!("Failed to create ISA builder: {}", e))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("Failed to create ISA: {}", e))?;

        // Create JIT builder with ISA
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register runtime symbols from plugins
        for (name, ptr) in symbols {
            builder.symbol(*name, *ptr);
        }

        // Create JIT module
        let mut module = JITModule::new(builder);
        let ctx = module.make_context();

        // Get target pointer type from ISA
        let pointer_type = module.target_config().pointer_type();

        Ok(Self {
            module,
            ctx,
            function_map: HashMap::new(),
            value_map: HashMap::new(),
            closure_environments: HashMap::new(),
            runtime_functions: HashMap::new(),
            pointer_type,
            module_counter: 0,
            defined_functions: HashSet::new(),
            current_env_param: None,
            string_data: HashMap::new(),
            string_counter: 0,
            qualified_name_to_func: HashMap::new(),
        })
    }

    /// Get the pointer size in bytes for the target architecture
    pub fn get_pointer_size(&self) -> u32 {
        match self.pointer_type {
            types::I32 => 4,
            types::I64 => 8,
            _ => 8, // Default to 64-bit
        }
    }

    /// Get the size of a type in bytes according to the target architecture
    pub fn get_type_size(&self, ty: &IrType) -> u64 {
        match ty {
            IrType::Void => 0,
            IrType::Bool | IrType::I8 | IrType::U8 => 1,
            IrType::I16 | IrType::U16 => 2,
            IrType::I32 | IrType::U32 | IrType::F32 => 4,
            IrType::I64 | IrType::U64 | IrType::F64 => 8,
            IrType::Ptr(_) | IrType::Ref(_) | IrType::Function { .. } => {
                self.get_pointer_size() as u64
            }
            IrType::Array(elem_ty, count) => self.get_type_size(elem_ty) * (*count as u64),
            IrType::Slice(_) | IrType::String => {
                // Slice is {ptr, len} = pointer + i64
                self.get_pointer_size() as u64 + 8
            }
            IrType::Struct { fields, .. } => {
                // Sum of field sizes (simplified, doesn't account for padding)
                fields.iter().map(|f| self.get_type_size(&f.ty)).sum()
            }
            IrType::Union { variants, .. } => {
                // Tag (i32) + max variant size
                let max_variant_size = variants
                    .iter()
                    .map(|v| v.fields.iter().map(|f| f.size()).sum::<usize>())
                    .max()
                    .unwrap_or(0);
                4 + max_variant_size as u64
            }
            IrType::Opaque { size, .. } => *size as u64,
            IrType::Any => {
                // {i64 type_id, ptr value_ptr}
                8 + self.get_pointer_size() as u64
            }
            IrType::TypeVar(_) => 8, // Safety net: pointer-sized if TypeVar leaks through
            IrType::Generic { .. } => 0, // Should be monomorphized before codegen
            // SIMD vector types - size is element_size * count
            IrType::Vector { element, count } => self.get_type_size(element) * (*count as u64),
        }
    }

    /// Get the alignment of a type in bytes (simplified)
    pub fn get_type_alignment(&self, ty: &IrType) -> u32 {
        match ty {
            IrType::Void => 1,
            IrType::Bool | IrType::I8 | IrType::U8 => 1,
            IrType::I16 | IrType::U16 => 2,
            IrType::I32 | IrType::U32 | IrType::F32 => 4,
            IrType::I64 | IrType::U64 | IrType::F64 => 8,
            IrType::Ptr(_) | IrType::Ref(_) | IrType::Function { .. } => self.get_pointer_size(),
            IrType::Array(elem_ty, _) => self.get_type_alignment(elem_ty),
            IrType::Slice(_) | IrType::String => self.get_pointer_size(), // Aligned to pointer
            IrType::Struct { fields, .. } => {
                // Max alignment of fields
                fields
                    .iter()
                    .map(|f| self.get_type_alignment(&f.ty))
                    .max()
                    .unwrap_or(1)
            }
            IrType::Union { variants, .. } => {
                // Max alignment of variant fields
                let max_align = variants
                    .iter()
                    .flat_map(|v| v.fields.iter())
                    .map(|f| self.get_type_alignment(f))
                    .max()
                    .unwrap_or(1);
                max_align.max(4) // At least 4 for the tag
            }
            IrType::Opaque { align, .. } => *align as u32,
            IrType::Any => 8,            // Aligned to i64
            IrType::TypeVar(_) => 8,     // Safety net: pointer-aligned if TypeVar leaks through
            IrType::Generic { .. } => 1, // Should be monomorphized before codegen
            // SIMD vectors require alignment matching their full size
            // 128-bit vectors need 16-byte alignment, 256-bit need 32-byte
            IrType::Vector { element, count } => {
                let size = self.get_type_size(ty) as u32;
                // Use vector size as alignment (16 for 128-bit, 32 for 256-bit)
                // but at minimum match element alignment
                let elem_align = self.get_type_alignment(element);
                size.max(elem_align)
            }
        }
    }

    /// Find the source function ID for a function pointer register
    /// This scans the MIR instructions to find if this register comes from FunctionRef or MakeClosure
    fn find_function_ref_source(func_ptr: IrId, function: &IrFunction) -> Option<IrFunctionId> {
        // First, try to find the direct source
        for (_, block) in &function.cfg.blocks {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::FunctionRef { dest, func_id } if *dest == func_ptr => {
                        return Some(*func_id);
                    }
                    IrInstruction::MakeClosure { dest, func_id, .. } if *dest == func_ptr => {
                        return Some(*func_id);
                    }
                    _ => {}
                }
            }
        }

        // If not found, check if func_ptr comes from a Load of a closure object
        // This handles the case where closure fn_ptr is extracted via Load
        for (_, block) in &function.cfg.blocks {
            for inst in &block.instructions {
                if let IrInstruction::Load { dest, ptr, .. } = inst {
                    if *dest == func_ptr {
                        // Check if ptr comes from a MakeClosure or PtrAdd of a MakeClosure
                        // First try direct MakeClosure
                        for (_, inner_block) in &function.cfg.blocks {
                            for inner_inst in &inner_block.instructions {
                                match inner_inst {
                                    IrInstruction::MakeClosure {
                                        dest: closure_dest,
                                        func_id,
                                        ..
                                    } if closure_dest == ptr => {
                                        debug!("Traced func_ptr through Load from MakeClosure to lambda {:?}", func_id);
                                        return Some(*func_id);
                                    }
                                    // Also check PtrAdd (for field access at offset)
                                    IrInstruction::PtrAdd {
                                        dest: ptr_add_dest,
                                        ptr: base_ptr,
                                        ..
                                    } if ptr_add_dest == ptr => {
                                        // Check if base_ptr is from MakeClosure
                                        for (_, deepest_block) in &function.cfg.blocks {
                                            for deepest_inst in &deepest_block.instructions {
                                                if let IrInstruction::MakeClosure {
                                                    dest: closure_dest,
                                                    func_id,
                                                    ..
                                                } = deepest_inst
                                                {
                                                    if closure_dest == base_ptr {
                                                        debug!("Traced func_ptr through Load->PtrAdd->MakeClosure to lambda {:?}", func_id);
                                                        return Some(*func_id);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Compile an entire MIR module
    pub fn compile_module(&mut self, mir_module: &IrModule) -> Result<(), String> {
        // Skip modules that only have extern functions (no implementations)
        // These are typically stdlib Haxe wrapper files (Thread.hx, Channel.hx, etc.)
        // that only declare externs. The actual implementations come from build_stdlib()
        // which gets merged into the user module.
        let has_implementations = mir_module
            .functions
            .values()
            .any(|f| !f.cfg.blocks.is_empty());

        if !has_implementations {
            debug!(
                ": Skipping module '{}' - no implementations (only {} extern declarations)",
                mir_module.name,
                mir_module.functions.len()
            );
            return Ok(());
        }

        // Increment module counter for unique function naming across modules
        // This prevents collisions when multiple MIR modules (each starting IDs from 0) are compiled
        let current_module = self.module_counter;
        self.module_counter += 1;

        // IMPORTANT: Don't clear function_map between modules!
        //
        // Each MIR module starts function IDs from 0, which would normally cause collisions.
        // However, we handle this with:
        // 1. Renumbered stdlib function IDs (done in compilation.rs) - ensures no ID collisions
        // 2. Unique function naming for regular functions (module_counter prefix)
        // 3. Extern function reuse (runtime_functions tracking) - prevents duplicate declarations
        //
        // We MUST NOT clear function_map because:
        // - Extern functions are shared across modules and need persistent MIR ID -> Cranelift ID mappings
        // - Without persistent mappings, Module 1's code can't call externs after Module 2 clears the map
        //
        // Previously we cleared function_map, which broke cross-module extern function references.
        debug!(
            ": Compiling module '{}' #{} (function_map has {} entries)",
            mir_module.name,
            current_module,
            self.function_map.len()
        );

        // First pass: declare all functions (except malloc/realloc/free which we handle separately)
        for (func_id, function) in &mir_module.functions {
            // Skip libc memory management functions - we'll declare them separately and map MIR IDs to libc
            if function.name == "malloc" || function.name == "realloc" || function.name == "free" {
                continue;
            }
            self.declare_function(*func_id, function)?;
        }

        // Declare memory management functions ONCE (across ALL modules)
        // Use libc malloc/free for best performance (tracked allocator was for debugging)
        if !self.runtime_functions.contains_key("malloc") {
            debug!("Declaring runtime function: malloc");
            self.declare_libc_function("malloc", 1, true)?;
        }
        if !self.runtime_functions.contains_key("realloc") {
            debug!("Declaring runtime function: realloc");
            self.declare_libc_function("realloc", 2, true)?;
        }
        if !self.runtime_functions.contains_key("free") {
            debug!("Declaring runtime function: free");
            self.declare_libc_function("free", 1, false)?;
        }

        // Declare rayzor_global_load and rayzor_global_store runtime functions
        // These are used by LoadGlobal and StoreGlobal instructions for static class fields
        if !self.runtime_functions.contains_key("rayzor_global_load") {
            debug!("Declaring runtime function: rayzor_global_load");
            self.declare_runtime_function("rayzor_global_load", &[types::I64], Some(types::I64))?;
        }
        if !self.runtime_functions.contains_key("rayzor_global_store") {
            debug!("Declaring runtime function: rayzor_global_store");
            self.declare_runtime_function("rayzor_global_store", &[types::I64, types::I64], None)?;
        }

        // Map MIR function IDs for malloc/realloc/free to their tracked Cranelift IDs
        // This ensures that when MIR code calls these functions, they resolve to tracked versions
        // Check both functions and extern_functions since malloc may be in either location
        for (func_id, function) in &mir_module.functions {
            if function.name == "malloc" {
                let libc_id = *self.runtime_functions.get("malloc").unwrap();
                debug!(
                    ": Mapping MIR malloc {:?} -> Cranelift {:?}",
                    func_id, libc_id
                );
                self.function_map.insert(*func_id, libc_id);
            } else if function.name == "realloc" {
                let libc_id = *self.runtime_functions.get("realloc").unwrap();
                debug!(
                    ": Mapping MIR realloc {:?} -> Cranelift {:?}",
                    func_id, libc_id
                );
                self.function_map.insert(*func_id, libc_id);
            } else if function.name == "free" {
                let libc_id = *self.runtime_functions.get("free").unwrap();
                debug!(
                    ": Mapping MIR free {:?} -> Cranelift {:?}",
                    func_id, libc_id
                );
                self.function_map.insert(*func_id, libc_id);
            }
        }

        // Also check extern_functions for malloc/realloc/free since heap allocation
        // now declares malloc as an extern function for proper linking
        for (func_id, extern_func) in &mir_module.extern_functions {
            if extern_func.name == "malloc" {
                let libc_id = *self.runtime_functions.get("malloc").unwrap();
                debug!(
                    ": Mapping MIR extern malloc {:?} -> Cranelift {:?}",
                    func_id, libc_id
                );
                self.function_map.insert(*func_id, libc_id);
            } else if extern_func.name == "realloc" {
                let libc_id = *self.runtime_functions.get("realloc").unwrap();
                debug!(
                    ": Mapping MIR extern realloc {:?} -> Cranelift {:?}",
                    func_id, libc_id
                );
                self.function_map.insert(*func_id, libc_id);
            } else if extern_func.name == "free" {
                let libc_id = *self.runtime_functions.get("free").unwrap();
                debug!(
                    ": Mapping MIR extern free {:?} -> Cranelift {:?}",
                    func_id, libc_id
                );
                self.function_map.insert(*func_id, libc_id);
            }
        }

        // Second pass: compile function bodies (skip extern functions with empty CFGs)
        for (func_id, function) in &mir_module.functions {
            // Skip extern functions (empty CFG means extern declaration)
            if function.cfg.blocks.is_empty() {
                debug!("Skipping extern function: {}", function.name);
                continue;
            }
            match self.compile_function(*func_id, mir_module, function) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!(
                        "[WARN] Skipping function '{}' ({}): {}",
                        function.name, func_id, e
                    );
                    // Define a trap stub so finalize_definitions doesn't panic
                    if let Err(e2) = self.define_trap_stub(*func_id, function) {
                        eprintln!(
                            "[WARN] Failed to define trap stub for '{}': {}",
                            function.name, e2
                        );
                    }
                }
            }
        }

        // Finalize the module
        self.module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize definitions: {}", e))?;

        Ok(())
    }

    /// Compile an entire MIR module WITHOUT calling finalize_definitions.
    ///
    /// This is used when compiling multiple modules to the same backend.
    /// Call `finalize()` after all modules are compiled.
    pub fn compile_module_without_finalize(&mut self, mir_module: &IrModule) -> Result<(), String> {
        // Skip modules that only have extern functions (no implementations)
        let has_implementations = mir_module
            .functions
            .values()
            .any(|f| !f.cfg.blocks.is_empty());

        if !has_implementations {
            debug!(
                ": Skipping module '{}' - no implementations (only {} extern declarations)",
                mir_module.name,
                mir_module.functions.len()
            );
            return Ok(());
        }

        // Increment module counter for unique function naming across modules
        let current_module = self.module_counter;
        self.module_counter += 1;

        debug!(
            ": Compiling module '{}' #{} (function_map has {} entries)",
            mir_module.name,
            current_module,
            self.function_map.len()
        );

        // First pass: declare all functions (except malloc/realloc/free which we handle separately)
        for (func_id, function) in &mir_module.functions {
            if function.name == "malloc" || function.name == "realloc" || function.name == "free" {
                continue;
            }
            self.declare_function(*func_id, function)?;
        }

        // Declare memory management functions ONCE (across ALL modules)
        // Use libc malloc/free for best performance
        if !self.runtime_functions.contains_key("malloc") {
            debug!("Declaring runtime function: malloc");
            self.declare_libc_function("malloc", 1, true)?;
        }
        if !self.runtime_functions.contains_key("realloc") {
            debug!("Declaring runtime function: realloc");
            self.declare_libc_function("realloc", 2, true)?;
        }
        if !self.runtime_functions.contains_key("free") {
            debug!("Declaring runtime function: free");
            self.declare_libc_function("free", 1, false)?;
        }

        // Declare rayzor_global_load and rayzor_global_store runtime functions
        if !self.runtime_functions.contains_key("rayzor_global_load") {
            debug!("Declaring runtime function: rayzor_global_load");
            self.declare_runtime_function("rayzor_global_load", &[types::I64], Some(types::I64))?;
        }
        if !self.runtime_functions.contains_key("rayzor_global_store") {
            debug!("Declaring runtime function: rayzor_global_store");
            self.declare_runtime_function("rayzor_global_store", &[types::I64, types::I64], None)?;
        }

        // Map MIR function IDs for malloc/realloc/free to their libc Cranelift IDs
        for (func_id, function) in &mir_module.functions {
            if function.name == "malloc" {
                let libc_id = *self.runtime_functions.get("malloc").unwrap();
                self.function_map.insert(*func_id, libc_id);
            } else if function.name == "realloc" {
                let libc_id = *self.runtime_functions.get("realloc").unwrap();
                self.function_map.insert(*func_id, libc_id);
            } else if function.name == "free" {
                let libc_id = *self.runtime_functions.get("free").unwrap();
                self.function_map.insert(*func_id, libc_id);
            }
        }

        // Also check extern_functions for malloc/realloc/free
        for (func_id, extern_func) in &mir_module.extern_functions {
            if extern_func.name == "malloc" {
                let libc_id = *self.runtime_functions.get("malloc").unwrap();
                self.function_map.insert(*func_id, libc_id);
            } else if extern_func.name == "realloc" {
                let libc_id = *self.runtime_functions.get("realloc").unwrap();
                self.function_map.insert(*func_id, libc_id);
            } else if extern_func.name == "free" {
                let libc_id = *self.runtime_functions.get("free").unwrap();
                self.function_map.insert(*func_id, libc_id);
            }
        }

        // Second pass: compile function bodies (skip extern functions with empty CFGs)
        for (func_id, function) in &mir_module.functions {
            if function.cfg.blocks.is_empty() {
                debug!("Skipping extern function: {}", function.name);
                continue;
            }
            match self.compile_function(*func_id, mir_module, function) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!(
                        "[WARN] Skipping function '{}' ({}): {}",
                        function.name, func_id, e
                    );
                    if let Err(e2) = self.define_trap_stub(*func_id, function) {
                        eprintln!(
                            "[WARN] Failed to define trap stub for '{}': {}",
                            function.name, e2
                        );
                    }
                }
            }
        }

        // NOTE: Do NOT call finalize_definitions() here - caller must call finalize() after all modules
        Ok(())
    }

    /// Finalize all compiled modules.
    ///
    /// This must be called after all `compile_module_without_finalize` calls are complete.
    /// After finalization, function pointers can be retrieved via `get_function_ptr`.
    pub fn finalize(&mut self) -> Result<(), String> {
        self.module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize definitions: {}", e))
    }

    /// Declare all functions from a module WITHOUT compiling their bodies.
    ///
    /// This is used by tiered compilation to prepare the backend for single-function
    /// recompilation. All functions must be declared first so that cross-function
    /// references can be resolved during compilation.
    ///
    /// Call this for ALL modules before calling `compile_single_function`.
    pub fn declare_module_functions(&mut self, mir_module: &IrModule) -> Result<(), String> {
        // Declare all functions (except malloc/realloc/free which we handle separately)
        for (func_id, function) in &mir_module.functions {
            if function.name == "malloc" || function.name == "realloc" || function.name == "free" {
                continue;
            }
            self.declare_function(*func_id, function)?;
        }

        // Declare C standard library memory functions ONCE (across ALL modules)
        if !self.runtime_functions.contains_key("malloc") {
            self.declare_libc_function("malloc", 1, true)?;
        }
        if !self.runtime_functions.contains_key("realloc") {
            self.declare_libc_function("realloc", 2, true)?;
        }
        if !self.runtime_functions.contains_key("free") {
            self.declare_libc_function("free", 1, false)?;
        }

        // Map MIR function IDs for malloc/realloc/free to their libc Cranelift IDs
        for (func_id, function) in &mir_module.functions {
            if function.name == "malloc" {
                let libc_id = *self.runtime_functions.get("malloc").unwrap();
                self.function_map.insert(*func_id, libc_id);
            } else if function.name == "realloc" {
                let libc_id = *self.runtime_functions.get("realloc").unwrap();
                self.function_map.insert(*func_id, libc_id);
            } else if function.name == "free" {
                let libc_id = *self.runtime_functions.get("free").unwrap();
                self.function_map.insert(*func_id, libc_id);
            }
        }

        // Also check extern_functions for malloc/realloc/free
        for (func_id, extern_func) in &mir_module.extern_functions {
            if extern_func.name == "malloc" {
                let libc_id = *self.runtime_functions.get("malloc").unwrap();
                self.function_map.insert(*func_id, libc_id);
            } else if extern_func.name == "realloc" {
                let libc_id = *self.runtime_functions.get("realloc").unwrap();
                self.function_map.insert(*func_id, libc_id);
            } else if extern_func.name == "free" {
                let libc_id = *self.runtime_functions.get("free").unwrap();
                self.function_map.insert(*func_id, libc_id);
            }
        }

        Ok(())
    }

    /// Compile a single function (for tiered compilation)
    ///
    /// This method declares, compiles, and finalizes a single function.
    /// Used by the tiered backend to recompile hot functions at higher optimization levels.
    pub fn compile_single_function(
        &mut self,
        mir_func_id: IrFunctionId,
        mir_module: &IrModule,
        function: &IrFunction,
    ) -> Result<(), String> {
        // Declare the function (if not already declared)
        if !self.function_map.contains_key(&mir_func_id) {
            self.declare_function(mir_func_id, function)?;
        }

        // Compile the function body
        self.compile_function(mir_func_id, mir_module, function)?;

        // Finalize this function
        self.module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize function: {}", e))?;

        Ok(())
    }

    /// Declare a function signature (first pass)
    fn declare_function(
        &mut self,
        mir_func_id: IrFunctionId,
        function: &IrFunction,
    ) -> Result<(), String> {
        // Determine if this is an extern function (empty CFG)
        let is_extern = function.cfg.blocks.is_empty();

        // CRITICAL: Check if this function was already declared (by name)
        // Both extern functions AND stdlib wrapper functions are shared across modules,
        // so we must not declare them twice!
        // We use runtime_functions to track all such functions (not just libc ones)
        if let Some(&existing_func_id) = self.runtime_functions.get(&function.name) {
            self.function_map.insert(mir_func_id, existing_func_id);
            return Ok(());
        }

        // CROSS-MODULE LINKING: Check if this function's qualified name matches a forward reference
        // Forward references are created when compiling cross-module calls (e.g., StringIteratorUnicode
        // calling StringTools.unsafeCodeAt before StringTools is fully compiled).
        // Forward refs use qualified name as their function name (e.g., "StringTools.unsafeCodeAt")
        if let Some(ref qualified_name) = function.qualified_name {
            // Check if there's a forward reference with this qualified name
            if let Some(&forward_ref_func_id) = self.runtime_functions.get(qualified_name) {
                // Only link if this function has a body (non-extern)
                // This allows the real implementation to use the forward ref's func_id
                if !is_extern {
                    debug!(
            ": Linking function '{}' to forward reference '{}' - MIR {:?} -> Cranelift {:?}",
                        function.name, qualified_name, mir_func_id, forward_ref_func_id
                    );
                    self.function_map.insert(mir_func_id, forward_ref_func_id);
                    // Also track by qualified name for future lookups
                    self.qualified_name_to_func
                        .insert(qualified_name.clone(), forward_ref_func_id);
                    return Ok(());
                }
            }
        }

        // Build Cranelift signature
        let mut sig = self.module.make_signature();

        // Check if we need sret (struct return by pointer)
        // MUST use sret for ALL functions (including extern) that return structs
        // because the C ABI on ARM64 uses sret for structs > 16 bytes
        let use_sret_in_signature = function.signature.uses_sret;

        if use_sret_in_signature {
            // Add sret parameter as first parameter
            sig.params.push(AbiParam::special(
                types::I64, // pointer type
                ArgumentPurpose::StructReturn,
            ));
        }

        // Add environment parameter (hidden first/second parameter) for non-extern functions
        // that use Haxe calling convention. All such functions accept an environment pointer
        // (null for static functions).
        // DO NOT add env parameter for:
        // - Extern functions (they're C ABI)
        // - C calling convention functions (they're wrappers around externs)
        // - Lambda functions (they already have an explicit 'env' parameter)
        let already_has_env_param = !function.signature.parameters.is_empty()
            && function.signature.parameters[0].name == "env";
        let is_c_calling_conv =
            function.signature.calling_convention == crate::ir::CallingConvention::C;

        if !is_extern && !already_has_env_param && !is_c_calling_conv {
            sig.params.push(AbiParam::new(types::I64));
        }

        // Add parameters
        for param in &function.signature.parameters {
            // For C calling convention extern functions on non-Windows platforms,
            // the ABI requires integer types smaller than 64 bits to be extended to i64.
            // On ARM64/AArch64 (Apple Silicon), i32 parameters are passed as i64.
            // This includes Bool (i8) since C ABI promotes all small integers.
            let will_extend = is_extern
                && function.signature.calling_convention == crate::ir::CallingConvention::C
                && !cfg!(target_os = "windows")
                && matches!(
                    param.ty,
                    crate::ir::IrType::I32
                        | crate::ir::IrType::U32
                        | crate::ir::IrType::Bool
                        | crate::ir::IrType::I8
                );

            if will_extend {
                debug!("!!! EXTENDING {} param '{}' from {:?} to i64 (is_extern={}, calling_conv={:?})",
                         function.name, param.name, param.ty, is_extern, function.signature.calling_convention);
            }

            let cranelift_type = if will_extend {
                types::I64
            } else {
                self.mir_type_to_cranelift(&param.ty)?
            };

            sig.params.push(AbiParam::new(cranelift_type));
        }

        // Debug: log Thread_spawn and Thread_join and channel extern signatures
        if function.name == "Thread_spawn"
            || function.name == "Thread_join"
            || function.name.starts_with("<lambda_")
            || function.name.starts_with("rayzor_channel")
            || function.name == "Channel_init"
        {
            debug!(
                ": Declaring '{}' (MIR {:?}) with {} params, is_extern={}, calling_conv={:?}",
                function.name,
                mir_func_id,
                function.signature.parameters.len(),
                is_extern,
                function.signature.calling_convention
            );
            for (i, param) in function.signature.parameters.iter().enumerate() {
                let cranelift_ty = self
                    .mir_type_to_cranelift(&param.ty)
                    .unwrap_or(types::INVALID);
                let actual_ty = if is_extern
                    && function.signature.calling_convention == crate::ir::CallingConvention::C
                    && !cfg!(target_os = "windows")
                {
                    match &param.ty {
                        crate::ir::IrType::I32
                        | crate::ir::IrType::U32
                        | crate::ir::IrType::Bool
                        | crate::ir::IrType::I8 => types::I64,
                        _ => cranelift_ty,
                    }
                } else {
                    cranelift_ty
                };
                debug!(
                    "  param[{}]: {} (MIR {:?} -> Cranelift {:?} -> actual {:?})",
                    i, param.name, param.ty, cranelift_ty, actual_ty
                );
            }
            debug!(
                "  return_type: {:?}, uses_sret: {}",
                function.signature.return_type, use_sret_in_signature
            );
        }

        // Debug: log lambda function signatures
        if function.name.starts_with("<lambda_") {
            debug!(
                " Lambda signature for {}: {} params",
                function.name,
                function.signature.parameters.len()
            );
            for (i, param) in function.signature.parameters.iter().enumerate() {
                trace!("  param{}: {} ({:?})", i, param.name, param.ty);
            }
        }

        // Add return type (unless using sret)
        if !use_sret_in_signature {
            let return_type = self.mir_type_to_cranelift(&function.signature.return_type)?;
            if return_type != types::INVALID {
                sig.returns.push(AbiParam::new(return_type));
            }
        }

        // Determine linkage and name based on whether this is an extern function
        let is_extern = function.cfg.blocks.is_empty();
        let (func_name, linkage) = if is_extern {
            // Extern functions use their actual name and Import linkage
            (function.name.clone(), Linkage::Import)
        } else {
            // Check if this is a stdlib MIR wrapper function by looking it up in the runtime mapping
            // Stdlib wrappers are functions registered in the runtime mapping system
            let stdlib_mapping = crate::stdlib::runtime_mapping::StdlibMapping::new();
            let is_stdlib_mir_wrapper = stdlib_mapping
                .find_by_runtime_name(&function.name)
                .is_some();

            if is_stdlib_mir_wrapper {
                // Stdlib MIR wrappers use their actual names with Export linkage
                // so that forward references can resolve to them
                (function.name.clone(), Linkage::Export)
            } else {
                // Regular functions get unique names and Export linkage
                // Include module_counter to avoid collisions when compiling multiple MIR modules
                if let Some(ref qualified_name) = function.qualified_name {
                    // Use qualified name for better debugging/profiling
                    (
                        format!(
                            "m{}__{}__func_{}",
                            self.module_counter,
                            qualified_name.replace(".", "_"),
                            mir_func_id.0
                        ),
                        Linkage::Export,
                    )
                } else {
                    (
                        format!("m{}_func_{}", self.module_counter, mir_func_id.0),
                        Linkage::Export,
                    )
                }
            }
        };

        let func_id = self
            .module
            .declare_function(&func_name, linkage, &sig)
            .map_err(|e| format!("Failed to declare function: {}", e))?;

        debug!(
            " Cranelift: Declared '{}' - MIR={:?} -> Cranelift={:?}, {} params",
            func_name,
            mir_func_id,
            func_id,
            function.signature.parameters.len()
        );
        self.function_map.insert(mir_func_id, func_id);

        // Track extern functions and stdlib wrapper functions in runtime_functions
        // so we don't declare them twice across MIR modules.
        // DO NOT track all functions — unqualified names like "new", "get", "toString"
        // collide across classes, causing signature mismatches.
        if is_extern {
            self.runtime_functions
                .insert(function.name.clone(), func_id);
        } else {
            let stdlib_mapping = crate::stdlib::runtime_mapping::StdlibMapping::new();
            if stdlib_mapping
                .find_by_runtime_name(&function.name)
                .is_some()
            {
                self.runtime_functions
                    .insert(function.name.clone(), func_id);
            }
        }

        Ok(())
    }

    /// Declare a libc function (malloc, realloc, free)
    /// These are provided by the system C library
    fn declare_libc_function(
        &mut self,
        name: &str,
        param_count: usize,
        has_return: bool,
    ) -> Result<FuncId, String> {
        // Check if already declared
        if let Some(&func_id) = self.runtime_functions.get(name) {
            return Ok(func_id);
        }

        // Create signature for standard libc memory functions
        let mut sig = self.module.make_signature();

        match name {
            "malloc" => {
                // fn malloc(size: size_t) -> *void
                // size_t is pointer-sized (i64 on 64-bit, i32 on 32-bit)
                sig.params.push(AbiParam::new(self.pointer_type)); // size
                sig.returns.push(AbiParam::new(self.pointer_type)); // *void
            }
            "realloc" => {
                // fn realloc(ptr: *void, size: size_t) -> *void
                sig.params.push(AbiParam::new(self.pointer_type)); // ptr
                sig.params.push(AbiParam::new(self.pointer_type)); // size
                sig.returns.push(AbiParam::new(self.pointer_type)); // *void
            }
            "free" => {
                // fn free(ptr: *void)
                sig.params.push(AbiParam::new(self.pointer_type)); // ptr
                                                                   // no return value
            }
            _ => return Err(format!("Unknown libc function: {}", name)),
        }

        // Declare the function with Import linkage (external symbol from libc)
        let func_id = self
            .module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| format!("Failed to declare libc function {}: {}", name, e))?;

        debug!(
            ": Declared libc {} as Cranelift func_id: {:?}",
            name, func_id
        );
        self.runtime_functions.insert(name.to_string(), func_id);
        Ok(func_id)
    }

    /// Declare a runtime function from our Rayzor runtime library
    ///
    /// Unlike libc functions, these have explicit parameter and return types specified.
    fn declare_runtime_function(
        &mut self,
        name: &str,
        param_types: &[types::Type],
        return_type: Option<types::Type>,
    ) -> Result<FuncId, String> {
        // Check if already declared
        if let Some(&func_id) = self.runtime_functions.get(name) {
            return Ok(func_id);
        }

        // Create signature
        let mut sig = self.module.make_signature();
        for &param_type in param_types {
            sig.params.push(AbiParam::new(param_type));
        }
        if let Some(ret_type) = return_type {
            sig.returns.push(AbiParam::new(ret_type));
        }

        // Declare the function with Import linkage (external symbol from runtime)
        let func_id = self
            .module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| format!("Failed to declare runtime function {}: {}", name, e))?;

        debug!(
            ": Declared runtime {} as Cranelift func_id: {:?}",
            name, func_id
        );
        self.runtime_functions.insert(name.to_string(), func_id);
        Ok(func_id)
    }

    /// Define a minimal trap stub for a function that failed compilation.
    /// This prevents cranelift's finalize_definitions from panicking on
    /// declared-but-uncompiled functions.
    fn define_trap_stub(
        &mut self,
        mir_func_id: IrFunctionId,
        function: &IrFunction,
    ) -> Result<(), String> {
        let func_id = *self
            .function_map
            .get(&mir_func_id)
            .ok_or("Function not declared")?;

        if self.defined_functions.contains(&func_id) {
            return Ok(());
        }

        self.ctx.func.clear();
        self.ctx.func.signature = self.module.make_signature();

        // Replicate the exact same signature as declare_function would have created
        let uses_sret = function.signature.uses_sret;
        if uses_sret {
            self.ctx.func.signature.params.push(AbiParam::special(
                self.pointer_type,
                ArgumentPurpose::StructReturn,
            ));
        }

        let already_has_env_param = !function.signature.parameters.is_empty()
            && function.signature.parameters[0].name == "env";
        let is_c_calling_conv =
            function.signature.calling_convention == crate::ir::CallingConvention::C;

        if !already_has_env_param && !is_c_calling_conv {
            self.ctx
                .func
                .signature
                .params
                .push(AbiParam::new(types::I64));
        }

        for param in &function.signature.parameters {
            let cranelift_type = self.mir_type_to_cranelift(&param.ty)?;
            self.ctx
                .func
                .signature
                .params
                .push(AbiParam::new(cranelift_type));
        }

        if !uses_sret {
            let return_type = self.mir_type_to_cranelift(&function.signature.return_type)?;
            if return_type != types::INVALID {
                self.ctx
                    .func
                    .signature
                    .returns
                    .push(AbiParam::new(return_type));
            }
        }

        // Build a minimal function body that just traps
        let mut builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut builder_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);
        builder
            .ins()
            .trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
        builder.finalize();

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define trap stub: {}", e))?;
        self.defined_functions.insert(func_id);
        self.module.clear_context(&mut self.ctx);

        Ok(())
    }

    /// Compile a function body (second pass)
    fn compile_function(
        &mut self,
        mir_func_id: IrFunctionId,
        mir_module: &IrModule,
        function: &IrFunction,
    ) -> Result<(), String> {
        // Get the Cranelift function ID
        let func_id = *self
            .function_map
            .get(&mir_func_id)
            .ok_or("Function not declared")?;

        // CRITICAL: Check if this function's body was already defined
        // This can happen when functions are shared across modules (e.g., stdlib wrappers)
        // and the same function appears in multiple MIR modules
        // We track by Cranelift FuncId (not MIR name) because different modules can have
        // functions with the same MIR name (e.g., 'new') but different Cranelift symbols
        if self.defined_functions.contains(&func_id) {
            return Ok(());
        }

        // Clear context for new function
        self.ctx.func.clear();
        self.ctx.func.signature = self.module.make_signature();

        // Check if we need sret (struct return convention)
        let uses_sret = function.signature.uses_sret;

        // If using sret, add hidden first parameter for return value pointer
        if uses_sret {
            self.ctx.func.signature.params.push(AbiParam::special(
                self.pointer_type,
                ArgumentPurpose::StructReturn,
            ));
        }

        // Add environment parameter for Haxe calling convention functions only
        // DO NOT add env parameter for:
        // - C calling convention functions (they're wrappers around externs)
        // - Lambda functions (they already have an explicit 'env' parameter)
        let already_has_env_param = !function.signature.parameters.is_empty()
            && function.signature.parameters[0].name == "env";
        let is_c_calling_conv =
            function.signature.calling_convention == crate::ir::CallingConvention::C;

        if !already_has_env_param && !is_c_calling_conv {
            self.ctx
                .func
                .signature
                .params
                .push(AbiParam::new(types::I64));
        }

        // Add parameters to signature
        debug!(
            " Cranelift: Function '{}' has {} parameters",
            function.name,
            function.signature.parameters.len()
        );
        for (i, param) in function.signature.parameters.iter().enumerate() {
            debug!(" Cranelift:   param[{}]: {} ({})", i, param.name, param.ty);
            let cranelift_type = self.mir_type_to_cranelift(&param.ty)?;
            self.ctx
                .func
                .signature
                .params
                .push(AbiParam::new(cranelift_type));
        }

        // Add return type to signature (void for sret functions)
        if uses_sret {
            // sret functions return void - the value is written through the pointer
        } else {
            let return_type = self.mir_type_to_cranelift(&function.signature.return_type)?;
            if return_type != types::INVALID {
                self.ctx
                    .func
                    .signature
                    .returns
                    .push(AbiParam::new(return_type));
            }
        }

        // Build the function body using FunctionBuilder
        let mut builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut builder_ctx);

        // Clear value map for new function
        self.value_map.clear();

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        // Map function parameters to their Cranelift values
        let param_values = builder.block_params(entry_block).to_vec();

        // Determine if this function already has an explicit 'env' parameter (e.g., lambdas)
        let already_has_env_param = !function.signature.parameters.is_empty()
            && function.signature.parameters[0].name == "env";
        let is_c_calling_conv =
            function.signature.calling_convention == crate::ir::CallingConvention::C;

        // If using sret, first parameter is the return pointer
        // Environment parameter is next (if added implicitly, not for lambdas with explicit env)
        let sret_offset = if uses_sret { 1 } else { 0 };

        if already_has_env_param {
            // Lambda with explicit env parameter: parameters map directly
            // No hidden environment parameter was added in declare_function
            // param_values[sret_offset] is the first user parameter (which is 'env')
            for (i, param) in function.signature.parameters.iter().enumerate() {
                self.value_map
                    .insert(param.reg, param_values[i + sret_offset]);
            }
            // For lambdas, the env parameter is the first user parameter
            // current_env_param should point to it
            self.current_env_param = Some(param_values[sret_offset]);
        } else if is_c_calling_conv {
            // C calling convention function: no hidden environment parameter
            // Parameters map directly (after optional sret)
            for (i, param) in function.signature.parameters.iter().enumerate() {
                self.value_map
                    .insert(param.reg, param_values[i + sret_offset]);
            }
            // No env param for C functions
            self.current_env_param = None;
        } else {
            // Regular Haxe function with implicit hidden environment parameter
            let env_offset = sret_offset; // env param is at this index
            let param_offset = env_offset + 1; // user params start after env

            // Store environment parameter for ClosureEnv
            self.current_env_param = Some(param_values[env_offset]);

            for (i, param) in function.signature.parameters.iter().enumerate() {
                self.value_map
                    .insert(param.reg, param_values[i + param_offset]);
            }
        }

        // Store sret pointer for use in Return terminator
        let sret_ptr = if uses_sret {
            Some(param_values[0])
        } else {
            None
        };

        // Note: Don't seal entry block yet, we need to add instructions first

        // First pass: Create all Cranelift blocks for MIR blocks
        let mut block_map = std::collections::HashMap::new();
        // debug!("Cranelift: Function {} has {} blocks in CFG", function.name, function.cfg.blocks.len());
        for (mir_block_id, mir_block) in &function.cfg.blocks {
            // debug!("Cranelift:   Block {:?} has {} phi nodes, {} instructions",
            //  mir_block_id, mir_block.phi_nodes.len(), mir_block.instructions.len());
            // Skip entry block as we already created it
            if mir_block_id.is_entry() {
                block_map.insert(*mir_block_id, entry_block);
            } else {
                let cl_block = builder.create_block();
                block_map.insert(*mir_block_id, cl_block);
            }
        }

        // Second pass: Translate instructions for each block
        // Process blocks in reverse post-order (RPO) to ensure definitions are
        // visited before uses, which is critical after inlining creates blocks
        // with higher IDs in the middle of control flow.
        let blocks_to_process = {
            let mut rpo = Vec::new();
            let mut visited = std::collections::HashSet::new();
            let mut stack = vec![(function.cfg.entry_block, false)];
            while let Some((block_id, processed)) = stack.pop() {
                if processed {
                    if let Some(block) = function.cfg.blocks.get(&block_id) {
                        rpo.push((block_id, block));
                    }
                    continue;
                }
                if !visited.insert(block_id) {
                    continue;
                }
                stack.push((block_id, true));
                // Push successors in reverse order so they're processed in forward order
                let successors: Vec<IrBlockId> =
                    match function.cfg.blocks.get(&block_id).map(|b| &b.terminator) {
                        Some(IrTerminator::Branch { target }) => vec![*target],
                        Some(IrTerminator::CondBranch {
                            true_target,
                            false_target,
                            ..
                        }) => vec![*true_target, *false_target],
                        Some(IrTerminator::Switch { cases, default, .. }) => {
                            let mut targets: Vec<IrBlockId> =
                                cases.iter().map(|(_, t)| *t).collect();
                            targets.push(*default);
                            targets
                        }
                        _ => vec![],
                    };
                for succ in successors.into_iter().rev() {
                    if !visited.contains(&succ) {
                        stack.push((succ, false));
                    }
                }
            }
            rpo.reverse();
            // Add any unreachable blocks not visited by RPO
            for (block_id, block) in &function.cfg.blocks {
                if !visited.contains(block_id) {
                    rpo.push((*block_id, block));
                }
            }
            rpo
        };

        // Track which blocks have been translated
        let mut translated_blocks = std::collections::HashSet::new();

        for (mir_block_id, mir_block) in blocks_to_process {
            let cl_block = *block_map
                .get(&mir_block_id)
                .ok_or_else(|| format!("Block {:?} not found in block_map", mir_block_id))?;

            // Switch to this block (entry block is already active, but switch anyway for clarity)
            builder.switch_to_block(cl_block);

            // Translate phi nodes first
            // debug!("Cranelift: Block {:?} has {} phi nodes", mir_block_id, mir_block.phi_nodes.len());
            for phi_node in &mir_block.phi_nodes {
                // eprintln!("  Phi node: dest={:?}, ty={:?}", phi_node.dest, phi_node.ty);
                // eprintln!("  Incoming edges ({}):", phi_node.incoming.len());
                // for (from_block, value_id) in &phi_node.incoming {
                //     eprintln!("    from block {:?}: value {:?}", from_block, value_id);
                // }
                Self::translate_phi_node_static(
                    &mut self.value_map,
                    &mut builder,
                    phi_node,
                    &block_map,
                    &function.cfg,
                )?;
                // eprintln!("    After translation, value_map has {:?}", self.value_map.keys().collect::<Vec<_>>());
            }

            // Translate instructions
            for instruction in &mir_block.instructions {
                Self::translate_instruction(
                    &mut self.value_map,
                    &mut builder,
                    instruction,
                    function,
                    &self.function_map,
                    &mut self.runtime_functions,
                    mir_module,
                    &mut self.module,
                    &mut self.closure_environments,
                    self.current_env_param,
                    &mut self.string_data,
                    &mut self.string_counter,
                )?;
            }

            // Translate terminator
            // debug!("Cranelift: MIR terminator for block {:?}: {:?}", mir_block_id, mir_block.terminator);
            if let Err(e) = Self::translate_terminator_static(
                &mut self.value_map,
                &mut builder,
                &mir_block.terminator,
                &block_map,
                function,
                sret_ptr,
            ) {
                debug!(
                    "\n!!! Error translating terminator in function '{}' block {:?}: {}",
                    function.name, mir_block_id, e
                );
                trace!("=== Cranelift IR so far ===");
                debug!("{}", self.ctx.func.display());
                trace!("=== End IR ===\n");
                return Err(e);
            }

            translated_blocks.insert(mir_block_id);
        }

        // Third pass: Seal all blocks after all have been translated
        // This is crucial for loops - a block can only be sealed after all predecessors
        // have been processed (including back edges)
        for mir_block_id in translated_blocks {
            let cl_block = *block_map.get(&mir_block_id).unwrap();
            builder.seal_block(cl_block);
        }

        // Finalize the function
        builder.finalize();

        // Print Cranelift IR if debug mode
        if cfg!(debug_assertions) {
            debug!("\n=== Cranelift IR for {} ===", function.name);
            debug!("{}", self.ctx.func.display());
            trace!("=== End Cranelift IR ===\n");
        }

        // Verify the function before defining (debug builds only)
        // This catches IR errors early but adds compilation overhead
        #[cfg(debug_assertions)]
        if let Err(errors) = cranelift_codegen::verify_function(&self.ctx.func, self.module.isa()) {
            return Err(format!("Verifier errors in {}: {}", function.name, errors));
        }

        // Define the function in the module
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define function: {}", e))?;

        // Track that this function has been defined to prevent duplicate definitions
        self.defined_functions.insert(func_id);
        debug!(
            ": Successfully defined function '{}' (MIR {:?}, Cranelift {:?})",
            function.name, mir_func_id, func_id
        );

        // Clear the context for next function
        self.module.clear_context(&mut self.ctx);

        Ok(())
    }

    /// Collect phi node arguments when branching to a block
    /// This function also coerces value types if they don't match the expected phi parameter type
    fn collect_phi_args_with_coercion(
        value_map: &HashMap<IrId, Value>,
        function: &IrFunction,
        target_block: IrBlockId,
        from_block: IrBlockId,
        builder: &mut FunctionBuilder,
    ) -> Result<Vec<BlockArg>, String> {
        let target = function
            .cfg
            .blocks
            .get(&target_block)
            .ok_or_else(|| format!("Target block {:?} not found", target_block))?;

        let mut phi_args = Vec::new();

        // For each phi node in the target block, find the incoming value from our block
        for phi_node in &target.phi_nodes {
            // Find the incoming value for this phi from our current block
            let incoming_value = phi_node
                .incoming
                .iter()
                .find(|(block_id, _)| *block_id == from_block)
                .map(|(_, value_id)| value_id)
                .ok_or_else(|| {
                    format!(
                        "No incoming value for phi node {:?} from block {:?}",
                        phi_node.dest, from_block
                    )
                })?;

            // Look up the Cranelift value for this MIR value
            let cl_value = *value_map.get(incoming_value).ok_or_else(|| {
                format!(
                    "Value {:?} not found in value_map for phi incoming (phi dest={:?}, from_block={:?}, target_block={:?}, func={})",
                    incoming_value, phi_node.dest, from_block, target_block, function.name
                )
            })?;

            // Get the expected Cranelift type from the phi node's MIR type
            let expected_cl_type = match &phi_node.ty {
                crate::ir::IrType::I8 => types::I8,
                crate::ir::IrType::I16 => types::I16,
                crate::ir::IrType::I32 => types::I32,
                crate::ir::IrType::I64 => types::I64,
                crate::ir::IrType::F32 => types::F32,
                crate::ir::IrType::F64 => types::F64,
                crate::ir::IrType::Bool => types::I8,
                crate::ir::IrType::Ptr(_) => types::I64,
                crate::ir::IrType::Ref(_) => types::I64,
                _ => types::I64,
            };

            // Check if the actual value type matches expected; coerce if not
            let actual_type = builder.func.dfg.value_type(cl_value);
            let final_value = if actual_type != expected_cl_type {
                debug!(
                    ": Phi arg type mismatch for {:?}: actual {:?}, expected {:?}, coercing",
                    phi_node.dest, actual_type, expected_cl_type
                );
                // Coerce the value
                match (actual_type, expected_cl_type) {
                    // i64 -> i32 truncation
                    (types::I64, types::I32) => builder.ins().ireduce(types::I32, cl_value),
                    // i32 -> i64 extension
                    (types::I32, types::I64) => builder.ins().sextend(types::I64, cl_value),
                    // i8 -> i32/i64
                    (types::I8, types::I32) => builder.ins().sextend(types::I32, cl_value),
                    (types::I8, types::I64) => builder.ins().sextend(types::I64, cl_value),
                    // Same type - no conversion needed
                    (from, to) if from == to => cl_value,
                    // Fallback: log warning and use as-is (may cause verifier error)
                    _ => {
                        debug!(
                            "WARNING: Cannot coerce phi arg from {:?} to {:?}",
                            actual_type, expected_cl_type
                        );
                        cl_value
                    }
                }
            } else {
                cl_value
            };

            // Wrap in BlockArg for the fork's phi node API
            phi_args.push(BlockArg::Value(final_value));
        }

        Ok(phi_args)
    }

    /// Translate a phi node to Cranelift IR
    fn translate_phi_node_static(
        value_map: &mut HashMap<IrId, Value>,
        builder: &mut FunctionBuilder,
        phi_node: &crate::ir::IrPhiNode,
        block_map: &HashMap<IrBlockId, Block>,
        _cfg: &crate::ir::IrControlFlowGraph,
    ) -> Result<(), String> {
        // Create block parameters for the phi node
        // In Cranelift, phi nodes are represented as block parameters

        // Get the Cranelift type for the phi node
        // For static methods, we need to use a simple type mapping
        let cl_type = match &phi_node.ty {
            crate::ir::IrType::I8 => cranelift_codegen::ir::types::I8,
            crate::ir::IrType::I16 => cranelift_codegen::ir::types::I16,
            crate::ir::IrType::I32 => cranelift_codegen::ir::types::I32,
            crate::ir::IrType::I64 => cranelift_codegen::ir::types::I64,
            crate::ir::IrType::F32 => cranelift_codegen::ir::types::F32,
            crate::ir::IrType::F64 => cranelift_codegen::ir::types::F64,
            crate::ir::IrType::Bool => cranelift_codegen::ir::types::I8,
            crate::ir::IrType::Ptr(_) => cranelift_codegen::ir::types::I64, // Assume 64-bit pointers
            crate::ir::IrType::Ref(_) => cranelift_codegen::ir::types::I64, // Assume 64-bit refs
            _ => cranelift_codegen::ir::types::I64,                         // Default
        };

        // Get the current block
        let current_block = builder
            .current_block()
            .ok_or_else(|| "No current block for phi node".to_string())?;

        // Append a block parameter
        let block_param = builder.append_block_param(current_block, cl_type);

        // Map the phi node's destination register to the block parameter
        value_map.insert(phi_node.dest, block_param);

        Ok(())
    }

    /// Translate a single MIR instruction to Cranelift IR (static method)
    fn translate_instruction(
        value_map: &mut HashMap<IrId, Value>,
        builder: &mut FunctionBuilder,
        instruction: &IrInstruction,
        function: &IrFunction,
        function_map: &HashMap<IrFunctionId, FuncId>,
        runtime_functions: &mut HashMap<String, FuncId>,
        mir_module: &IrModule,
        module: &mut JITModule,
        closure_environments: &mut HashMap<IrId, Value>,
        current_env_param: Option<Value>,
        string_data: &mut HashMap<String, DataId>,
        string_counter: &mut usize,
    ) -> Result<(), String> {
        use crate::ir::IrInstruction;

        // Debug: Log every instruction being lowered
        if matches!(instruction, IrInstruction::Cast { .. }) {
            debug!("Cranelift: Lowering Cast instruction: {:?}", instruction);
        }

        match instruction {
            IrInstruction::Const { dest, value } => {
                let cl_value = Self::translate_const_value(
                    builder,
                    value,
                    function_map,
                    runtime_functions,
                    module,
                    string_data,
                    string_counter,
                )?;
                value_map.insert(*dest, cl_value);
            }

            IrInstruction::Copy { dest, src } => {
                // Copy: For Copy types (Int, Bool, etc.) - just copy the value
                let src_value = *value_map
                    .get(src)
                    .ok_or_else(|| format!("Source value {:?} not found", src))?;
                value_map.insert(*dest, src_value);
                // Note: src remains valid after copy
            }

            IrInstruction::Move { dest, src } => {
                // Move: Transfer ownership - move the value and invalidate source
                let src_value = *value_map
                    .get(src)
                    .ok_or_else(|| format!("Source value {:?} not found for move", src))?;
                value_map.insert(*dest, src_value);
                // Invalidate source - any future use is a compile error (caught by MIR validation)
                // In codegen, we just don't remove it from value_map to keep the value alive
                // The MIR validator ensures src isn't used after the move
            }

            IrInstruction::BorrowImmutable {
                dest,
                src,
                lifetime: _,
            } => {
                // Borrow immutable: Create a pointer to the value
                // In Cranelift, this is just the address of the value
                let src_value = *value_map
                    .get(src)
                    .ok_or_else(|| format!("Source value {:?} not found for borrow", src))?;

                // For heap-allocated objects, the value is already a pointer - just use it
                // For stack values, we'd need to take their address
                // TODO: Distinguish between stack and heap values
                value_map.insert(*dest, src_value);

                // Note: src remains valid - borrows don't invalidate
                // Multiple immutable borrows allowed (enforced by MIR validation)
            }

            IrInstruction::BorrowMutable {
                dest,
                src,
                lifetime: _,
            } => {
                // Borrow mutable: Create an exclusive pointer to the value
                let src_value = *value_map
                    .get(src)
                    .ok_or_else(|| format!("Source value {:?} not found for mut borrow", src))?;

                // Like immutable borrow, but exclusive
                value_map.insert(*dest, src_value);

                // Note: MIR validation ensures:
                // 1. Only ONE mutable borrow exists at a time
                // 2. No immutable borrows exist while mutable borrow is active
                // 3. src is not accessed while borrowed mutably
            }

            IrInstruction::Clone { dest, src } => {
                // Clone: Call the clone function for this type
                let src_value = *value_map
                    .get(src)
                    .ok_or_else(|| format!("Source value {:?} not found for clone", src))?;

                // Look up the clone function for this type
                // For now, we'll use memcpy for simple objects
                // TODO: Call actual clone() method if type has one

                // Get the size of the object to clone
                // For heap objects, we need to allocate new memory and deep copy
                let ptr_type = module.target_config().pointer_type();

                // Allocate new memory (same size as source)
                // This is simplified - real implementation needs type info
                let size_val = builder.ins().iconst(ptr_type, 64); // Placeholder size

                // Call rayzor_malloc - need to convert FuncId to FuncRef
                let malloc_func_id = *runtime_functions
                    .get("malloc") // Use libc malloc directly
                    .ok_or_else(|| "malloc not found".to_string())?;
                let malloc_func_ref = module.declare_func_in_func(malloc_func_id, builder.func);

                // Malloc is a libc function, it does NOT take an environment parameter.
                // The CallDirect logic below handles this distinction.
                let inst = builder.ins().call(malloc_func_ref, &[size_val]);
                let new_ptr = builder.inst_results(inst)[0];

                // Copy data from source to destination
                // TODO: Use actual memcpy or call clone() method
                builder.emit_small_memory_copy(
                    module.target_config(),
                    new_ptr,
                    src_value,
                    64, // size
                    8,  // alignment
                    8,  // alignment
                    true,
                    cranelift_codegen::ir::MemFlags::new(),
                );

                value_map.insert(*dest, new_ptr);
                // Both src and dest now own separate objects
            }

            IrInstruction::EndBorrow { borrow } => {
                // EndBorrow: Explicitly end a borrow's lifetime
                // In Cranelift, this is mostly a no-op since borrows are just pointers
                // The main purpose is to mark the end of the borrow scope for validation

                // We could optionally remove from value_map to catch use-after-end-borrow
                // but that's already enforced by MIR validation

                // In a more sophisticated implementation, we might:
                // 1. Insert debug assertions to check borrow validity
                // 2. Update borrow tracking metadata
                // 3. Enable optimizations (value can be moved after borrow ends)

                // For now, it's just a marker for the validator
            }

            IrInstruction::BinOp {
                dest,
                op,
                left,
                right,
            } => {
                // Get type from register_types map first, then fall back to locals
                let ty = function
                    .register_types
                    .get(dest)
                    .or_else(|| function.register_types.get(left))
                    .or_else(|| function.locals.get(dest).map(|local| &local.ty))
                    .ok_or_else(|| format!("Type not found for BinOp dest {:?}", dest))?;

                let value =
                    Self::lower_binary_op_static(value_map, builder, op, ty, *left, *right)?;
                value_map.insert(*dest, value);
            }

            IrInstruction::UnOp { dest, op, operand } => {
                // Get type from register_types map first, then fall back to locals
                let ty = function
                    .register_types
                    .get(dest)
                    .or_else(|| function.register_types.get(operand))
                    .or_else(|| function.locals.get(dest).map(|local| &local.ty))
                    .ok_or_else(|| format!("Type not found for UnOp dest {:?}", dest))?;

                let value = Self::lower_unary_op_static(value_map, builder, op, ty, *operand)?;
                value_map.insert(*dest, value);
            }

            IrInstruction::Cmp {
                dest,
                op,
                left,
                right,
            } => {
                // Get type from register_types map, locals, or phi nodes
                let phi_ty = function
                    .cfg
                    .blocks
                    .values()
                    .flat_map(|b| b.phi_nodes.iter())
                    .find(|phi| phi.dest == *left)
                    .map(|phi| &phi.ty);
                let ty = function
                    .register_types
                    .get(left)
                    .or_else(|| function.locals.get(left).map(|local| &local.ty))
                    .or(phi_ty)
                    .ok_or_else(|| format!("Type not found for Cmp operand {:?} (right={:?}, dest={:?}, op={:?}) in function '{}'", left, right, dest, op, function.name))?;

                let value =
                    Self::lower_compare_op_static(value_map, builder, op, ty, *left, *right)?;
                value_map.insert(*dest, value);
            }

            IrInstruction::Load { dest, ptr, ty } => {
                let value = Self::lower_load_static(value_map, builder, ty, *ptr)?;
                value_map.insert(*dest, value);
            }

            IrInstruction::Store { ptr, value } => {
                // Widen narrow integer values when storing to wide struct field slots.
                // All Rayzor object fields are 8-byte slots (elem_size = 8). When an
                // i32 value is stored to a GEP-derived pointer (struct field), only 4
                // bytes are written, leaving upper bytes uninitialized. This causes
                // corruption when the field is later loaded as i64 (generic type params).
                // We detect GEP targets via register_types: GEP results have Ptr(elem_ty).
                // Widen i32/i16/i8 stores to ALL struct field pointers (any Ptr type except Ptr(U8)/Ptr(I8)).
                // All Rayzor object field slots are 8 bytes, so we must store 8 bytes to avoid
                // leaving upper bytes uninitialized, which causes garbage reads when loaded as i32
                // from unzeroed malloc memory with adjacent pointer fields.
                let needs_widen = if let Some(ptr_ty) = function.register_types.get(ptr) {
                    if let IrType::Ptr(inner) = ptr_ty {
                        // Don't widen for byte-pointer stores (Ptr(U8)/Ptr(I8)) which are byte-addressed
                        !matches!(inner.as_ref(), IrType::U8 | IrType::I8)
                    } else {
                        false
                    }
                } else {
                    false
                };
                if needs_widen {
                    let val = *value_map.get(value).ok_or("Store: value not found")?;
                    let ptr_val = *value_map.get(ptr).ok_or("Store: ptr not found")?;
                    let val_ty = builder.func.dfg.value_type(val);
                    let val = if val_ty.bits() < 64 && val_ty.is_int() {
                        builder.ins().sextend(types::I64, val)
                    } else {
                        val
                    };
                    let flags = MemFlags::new().with_aligned().with_notrap();
                    builder.ins().store(flags, val, ptr_val, 0);
                } else {
                    Self::lower_store_static(value_map, builder, *ptr, *value)?;
                }
            }

            IrInstruction::Alloc { dest, ty, count } => {
                // Get count value - look for it in value_map
                let count_val = match count {
                    Some(count_id) => {
                        // Try to get the count from value_map (should be a Cranelift iconst)
                        if let Some(&count_value) = value_map.get(count_id) {
                            // Extract immediate value from Cranelift instruction
                            // The value should be defined by an iconst instruction
                            match builder.func.dfg.value_def(count_value) {
                                cranelift_codegen::ir::ValueDef::Result(inst, _) => {
                                    if let cranelift_codegen::ir::InstructionData::UnaryImm {
                                        imm,
                                        ..
                                    } = builder.func.dfg.insts[inst]
                                    {
                                        Some(imm.bits() as u32)
                                    } else {
                                        warn!("Alloc count instruction is not UnaryImm, defaulting to 1");
                                        Some(1)
                                    }
                                }
                                _ => {
                                    warn!("Alloc count value not from instruction result, defaulting to 1");
                                    Some(1)
                                }
                            }
                        } else {
                            warn!(
                                "Alloc count IrId {:?} not found in value_map, defaulting to 1",
                                count_id
                            );
                            Some(1)
                        }
                    }
                    None => None,
                };
                let value = Self::lower_alloca_static(builder, ty, count_val)?;
                value_map.insert(*dest, value);
            }

            IrInstruction::CallDirect {
                dest,
                func_id,
                args,
                arg_ownership: _,
                type_args: _, // Handled by monomorphization pass before codegen
                is_tail_call: _,
            } => {
                // TODO: Use arg_ownership to generate proper move/borrow/clone code
                // Check if this is an extern function call
                if let Some(extern_func) = mir_module.extern_functions.get(func_id) {
                    // This is an external runtime function call
                    info!("Calling external function: {}", extern_func.name);
                    debug!(
                        "[extern_func] calling_convention={:?}, param_count={}",
                        extern_func.signature.calling_convention,
                        extern_func.signature.parameters.len()
                    );
                    for (i, p) in extern_func.signature.parameters.iter().enumerate() {
                        debug!("[extern_func]   param[{}] '{}': {:?}", i, p.name, p.ty);
                    }

                    // Declare the external function if not already declared
                    let cl_func_id = if let Some(&id) = runtime_functions.get(&extern_func.name) {
                        debug!("[extern_func] Already declared as {:?}", id);
                        id
                    } else {
                        // Declare the external runtime function dynamically
                        let mut sig = module.make_signature();

                        // Add parameters using actual types from the extern function signature
                        // Apply C ABI integer promotion for non-Windows platforms
                        for param in &extern_func.signature.parameters {
                            let mut cranelift_type = Self::mir_type_to_cranelift_static(&param.ty)?;

                            // For C calling convention externs on non-Windows platforms, extend i32/u32 to i64
                            if !cfg!(target_os = "windows")
                                && extern_func.signature.calling_convention
                                    == crate::ir::CallingConvention::C
                            {
                                match param.ty {
                                    crate::ir::IrType::I32 | crate::ir::IrType::U32 => {
                                        debug!("!!! [DYNAMIC DECL] Extending {} param '{}' from {:?} to i64", extern_func.name, param.name, param.ty);
                                        cranelift_type = types::I64;
                                    }
                                    _ => {}
                                }
                            }

                            sig.params.push(AbiParam::new(cranelift_type));
                        }

                        // Add return type using actual type from the extern function signature
                        if extern_func.signature.return_type != crate::ir::IrType::Void {
                            let return_type = Self::mir_type_to_cranelift_static(
                                &extern_func.signature.return_type,
                            )?;
                            if return_type != types::INVALID {
                                sig.returns.push(AbiParam::new(return_type));
                            }
                        }

                        let id = module
                            .declare_function(&extern_func.name, Linkage::Import, &sig)
                            .map_err(|e| {
                                format!(
                                    "Failed to declare runtime function {}: {}",
                                    extern_func.name, e
                                )
                            })?;

                        info!(
                            "INFO: Declared external runtime function {} as func_id: {:?}",
                            extern_func.name, id
                        );
                        runtime_functions.insert(extern_func.name.clone(), id);
                        id
                    };

                    let func_ref = module.declare_func_in_func(cl_func_id, builder.func);

                    // Lower arguments
                    // For C extern functions on non-Windows platforms, extend i32/u32 to i64
                    let mut arg_values = Vec::new();

                    // Get the expected Cranelift parameter types from the function signature
                    let expected_sig = module.declarations().get_function_decl(cl_func_id);
                    let expected_param_types: Vec<_> = expected_sig
                        .signature
                        .params
                        .iter()
                        .map(|p| p.value_type)
                        .collect();

                    for (i, &arg_reg) in args.iter().enumerate() {
                        let mut cl_value = *value_map.get(&arg_reg).ok_or_else(|| {
                            format!("Argument register {:?} not found in value_map", arg_reg)
                        })?;

                        // Check if this C extern function parameter needs extension
                        // On ARM64/Apple Silicon, C ABI requires i32 values to be extended to i64
                        if !cfg!(target_os = "windows")
                            && extern_func.signature.calling_convention
                                == crate::ir::CallingConvention::C
                        {
                            // Get the actual Cranelift type of the value
                            let cl_value_type = builder.func.dfg.value_type(cl_value);

                            // Get the expected Cranelift type from the signature (not the MIR type)
                            let expected_cl_type = expected_param_types.get(i).copied();

                            // Extend i32 to i64 if the Cranelift signature expects i64
                            if cl_value_type == types::I32 && expected_cl_type == Some(types::I64) {
                                debug!("!!! [EXTERN BRANCH] Extending arg {} for {} from i32 to i64 (based on Cranelift signature)",
                                    i, extern_func.name);
                                cl_value = builder.ins().sextend(types::I64, cl_value);
                            } else if cl_value_type == types::I8
                                && expected_cl_type == Some(types::I64)
                            {
                                debug!(
                                    "!!! [EXTERN BRANCH] Extending arg {} for {} from i8 to i64",
                                    i, extern_func.name
                                );
                                cl_value = builder.ins().sextend(types::I64, cl_value);
                            } else if let Some(param) = extern_func.signature.parameters.get(i) {
                                // Fallback to MIR type-based conversions for special cases
                                if (cl_value_type == types::I32 || cl_value_type == types::I64)
                                    && matches!(param.ty, crate::ir::IrType::F64)
                                {
                                    // Integer to float conversion (e.g., Math.abs(-5))
                                    cl_value = builder.ins().fcvt_from_sint(types::F64, cl_value);
                                } else if (cl_value_type == types::I32
                                    || cl_value_type == types::I64)
                                    && matches!(param.ty, crate::ir::IrType::F32)
                                {
                                    // Integer to float32 conversion
                                    cl_value = builder.ins().fcvt_from_sint(types::F32, cl_value);
                                }
                            }
                        }
                        arg_values.push(cl_value);
                    }

                    // Check if this is a function we can replace with inline instructions
                    let intrinsic_result = match extern_func.name.as_str() {
                        // Math intrinsics: replace with native Cranelift instructions
                        "haxe_math_sqrt" | "haxe_math_abs" | "haxe_math_floor"
                        | "haxe_math_ceil" | "haxe_math_round"
                            if arg_values.len() == 1 =>
                        {
                            let arg = arg_values[0];
                            let arg_type = builder.func.dfg.value_type(arg);

                            // Convert int to float if necessary, or skip if unsupported type
                            if arg_type.is_float()
                                || arg_type == types::I32
                                || arg_type == types::I64
                            {
                                let float_arg = if arg_type.is_float() {
                                    arg
                                } else {
                                    builder.ins().fcvt_from_sint(types::F64, arg)
                                };

                                let result = match extern_func.name.as_str() {
                                    "haxe_math_sqrt" => builder.ins().sqrt(float_arg),
                                    "haxe_math_abs" => builder.ins().fabs(float_arg),
                                    "haxe_math_floor" => builder.ins().floor(float_arg),
                                    "haxe_math_ceil" => builder.ins().ceil(float_arg),
                                    "haxe_math_round" => builder.ins().nearest(float_arg),
                                    _ => unreachable!(),
                                };
                                debug!(
                                    "Math intrinsic: {} → native Cranelift instruction",
                                    extern_func.name
                                );
                                Some(result)
                            } else {
                                // Unknown type, skip intrinsic and fall through to function call
                                debug!(
                                    "Math intrinsic: {} skipped, unsupported arg type {:?}",
                                    extern_func.name, arg_type
                                );
                                None
                            }
                        }
                        // Std.int(): float to int truncation
                        "haxe_std_int" if arg_values.len() == 1 => {
                            let arg = arg_values[0];
                            let arg_type = builder.func.dfg.value_type(arg);
                            let result = if arg_type.is_float() {
                                // fcvt_to_sint_sat: saturating conversion (avoids UB on overflow)
                                builder.ins().fcvt_to_sint_sat(types::I32, arg)
                            } else if arg_type == types::I64 {
                                // Already an int, just narrow to i32
                                builder.ins().ireduce(types::I32, arg)
                            } else {
                                // Already i32, return as-is
                                arg
                            };
                            debug!("Std intrinsic: haxe_std_int → {:?} to i32", arg_type);
                            Some(result)
                        }
                        // Array intrinsics
                        // HaxeArray layout: { ptr: *mut u8 (0), len: usize (8), cap: usize (16), elem_size: usize (24) }
                        "haxe_array_length" if arg_values.len() == 1 => {
                            let arr_ptr = arg_values[0];
                            let len = builder.ins().load(
                                types::I64,
                                MemFlags::trusted(),
                                arr_ptr,
                                8i32, // offset of len field
                            );
                            debug!("Array intrinsic: haxe_array_length → inline load");
                            Some(len)
                        }
                        // Array get_ptr: compute ptr + index * elem_size
                        // Returns pointer to element at given index
                        "haxe_array_get_ptr" if arg_values.len() == 2 => {
                            let arr_ptr = arg_values[0];
                            let index = arg_values[1];
                            // Load base data pointer (offset 0)
                            let data_ptr =
                                builder
                                    .ins()
                                    .load(types::I64, MemFlags::trusted(), arr_ptr, 0i32);
                            // Load elem_size (offset 24)
                            let elem_size =
                                builder
                                    .ins()
                                    .load(types::I64, MemFlags::trusted(), arr_ptr, 24i32);
                            // Compute offset = index * elem_size
                            let byte_offset = builder.ins().imul(index, elem_size);
                            // Compute element pointer = data_ptr + offset
                            let elem_ptr = builder.ins().iadd(data_ptr, byte_offset);
                            debug!(
                                "Array intrinsic: haxe_array_get_ptr → inline pointer arithmetic"
                            );
                            Some(elem_ptr)
                        }
                        _ => None,
                    };

                    if let Some(result) = intrinsic_result {
                        if let Some(dest_reg) = dest {
                            value_map.insert(*dest_reg, result);
                        }
                    } else {
                        // Make the call
                        let call_inst = builder.ins().call(func_ref, &arg_values);

                        // Get return value if any
                        if let Some(dest_reg) = dest {
                            let results = builder.inst_results(call_inst);
                            if !results.is_empty() {
                                value_map.insert(*dest_reg, results[0]);
                            }
                        }
                    }
                } else {
                    // Check if this is a call to malloc/realloc/free
                    let called_func = mir_module.functions.get(func_id).ok_or_else(|| {
                        format!("Called function {:?} not found in module", func_id)
                    })?;

                    let (cl_func_id, func_ref) = if called_func.name == "malloc"
                        || called_func.name == "realloc"
                        || called_func.name == "free"
                    {
                        // This is a memory management function - call the libc version
                        let libc_id =
                            *runtime_functions.get(&called_func.name).ok_or_else(|| {
                                format!("libc function {} not declared", called_func.name)
                            })?;
                        debug!(
                            "[In {}] Redirecting {} call (MIR func_id={:?}) to libc func_id: {:?}",
                            function.name, called_func.name, func_id, libc_id
                        );
                        let func_ref = module.declare_func_in_func(libc_id, builder.func);
                        debug!(
                            ": [In {}] Got func_ref for {} (libc_id={:?}): {:?}",
                            function.name, called_func.name, libc_id, func_ref
                        );
                        (libc_id, func_ref)
                    } else {
                        // Normal MIR function call
                        let cl_func_id = *function_map.get(func_id).ok_or_else(|| {
                            format!("Function {:?} not found in function_map", func_id)
                        })?;
                        let func_ref = module.declare_func_in_func(cl_func_id, builder.func);
                        (cl_func_id, func_ref)
                    };

                    // Check if function uses sret (and is not extern)
                    let is_extern_func = called_func.cfg.blocks.is_empty();
                    let uses_sret = called_func.signature.uses_sret && !is_extern_func;

                    // Allocate stack space for sret if needed
                    let sret_slot = if uses_sret {
                        let ret_ty = &called_func.signature.return_type;
                        Some(Self::lower_alloca_static(builder, ret_ty, None)?)
                    } else {
                        None
                    };

                    // Translate arguments (prepend sret pointer if needed)
                    let mut call_args = Vec::new();
                    if let Some(sret_ptr) = sret_slot {
                        call_args.push(sret_ptr);
                    }

                    // Add environment argument (null for direct calls to Haxe calling convention functions)
                    // DO NOT add env argument for:
                    // - Extern functions (they're C ABI)
                    // - C calling convention functions (they're wrappers around externs)
                    // - Lambda functions (they already have an explicit 'env' parameter)
                    // - malloc/realloc/free (libc functions)
                    let is_lambda = called_func.name.starts_with("<lambda");
                    let is_c_calling_conv =
                        called_func.signature.calling_convention == crate::ir::CallingConvention::C;

                    let should_add_env = !is_extern_func
                        && !is_lambda
                        && !is_c_calling_conv
                        && !(called_func.name == "malloc"
                            || called_func.name == "realloc"
                            || called_func.name == "free");
                    if should_add_env {
                        // For direct calls to regular functions, we pass a null environment pointer
                        call_args.push(builder.ins().iconst(types::I64, 0));
                    }

                    // For C extern functions on non-Windows platforms, extend i32/u32 arguments to i64
                    // A function is C extern if:
                    // 1. It has C calling convention AND
                    // 2. Either (a) it has no blocks (true extern) OR (b) it has External linkage (wrapper around extern)
                    let is_c_extern = called_func.signature.calling_convention
                        == crate::ir::CallingConvention::C
                        && (is_extern_func
                            || called_func.attributes.linkage == crate::ir::Linkage::External)
                        && !cfg!(target_os = "windows");

                    for (i, arg_id) in args.iter().enumerate() {
                        let mut arg_val = *value_map.get(arg_id).ok_or_else(|| {
                            format!("Argument {:?} not found in value_map (in function '{}', calling '{}')", arg_id, function.name, called_func.name)
                        })?;

                        // Get the actual Cranelift type of the argument
                        let actual_cl_ty = builder.func.dfg.value_type(arg_val);

                        // For C extern functions on non-Windows, apply integer promotion FIRST
                        // This must happen before the MIR type comparison because Cranelift signature
                        // was declared with i64 params (C ABI promotion) but MIR still has I32 types.
                        // The key check is: if the actual Cranelift value is i32/i8 and the parameter
                        // expects i64, we need to extend.
                        if is_c_extern {
                            if let Some(param) = called_func.signature.parameters.get(i) {
                                // C ABI: extend small integer values to i64
                                // This covers cases where MIR declares I32 params (need extension)
                                // OR where MIR declares I64 but value is still i32 (also needs extension)
                                if actual_cl_ty == types::I32 || actual_cl_ty == types::I8 {
                                    match &param.ty {
                                        crate::ir::IrType::I32 | crate::ir::IrType::I64 => {
                                            // Sign-extend to i64 for C ABI
                                            arg_val = builder.ins().sextend(types::I64, arg_val);
                                        }
                                        crate::ir::IrType::U32 | crate::ir::IrType::U64 => {
                                            // Zero-extend to i64 for C ABI
                                            arg_val = builder.ins().uextend(types::I64, arg_val);
                                        }
                                        crate::ir::IrType::Bool => {
                                            // Bools are extended to i64 in C ABI
                                            arg_val = builder.ins().sextend(types::I64, arg_val);
                                        }
                                        crate::ir::IrType::F64 => {
                                            // Convert integer to f64 (e.g., Math.abs(-5))
                                            arg_val =
                                                builder.ins().fcvt_from_sint(types::F64, arg_val);
                                        }
                                        crate::ir::IrType::F32 => {
                                            // Convert integer to f32
                                            arg_val =
                                                builder.ins().fcvt_from_sint(types::F32, arg_val);
                                        }
                                        _ => {}
                                    }
                                } else if actual_cl_ty == types::I64 {
                                    // Also handle i64 -> f64 conversion
                                    match &param.ty {
                                        crate::ir::IrType::F64 => {
                                            arg_val =
                                                builder.ins().fcvt_from_sint(types::F64, arg_val);
                                        }
                                        crate::ir::IrType::F32 => {
                                            arg_val =
                                                builder.ins().fcvt_from_sint(types::F32, arg_val);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        } else {
                            // For non-C-extern functions, compare against MIR types and convert if needed
                            let expected_cl_ty =
                                if let Some(param) = called_func.signature.parameters.get(i) {
                                    Self::mir_type_to_cranelift_static(&param.ty)?
                                } else {
                                    types::I64 // Default fallback
                                };

                            // Insert type conversion if needed
                            if actual_cl_ty != expected_cl_ty {
                                if actual_cl_ty == types::I32 && expected_cl_ty == types::I64 {
                                    // Sign-extend i32 to i64
                                    arg_val = builder.ins().sextend(types::I64, arg_val);
                                } else if actual_cl_ty == types::I64 && expected_cl_ty == types::I32
                                {
                                    // Reduce i64 to i32
                                    arg_val = builder.ins().ireduce(types::I32, arg_val);
                                } else if (actual_cl_ty == types::I32 || actual_cl_ty == types::I64)
                                    && expected_cl_ty == types::F64
                                {
                                    // Integer to float conversion (e.g., Math.abs(-5))
                                    arg_val = builder.ins().fcvt_from_sint(types::F64, arg_val);
                                } else if (actual_cl_ty == types::I32 || actual_cl_ty == types::I64)
                                    && expected_cl_ty == types::F32
                                {
                                    // Integer to float32 conversion
                                    arg_val = builder.ins().fcvt_from_sint(types::F32, arg_val);
                                } else if actual_cl_ty == types::F64 && expected_cl_ty == types::I64
                                {
                                    // Float to integer conversion
                                    arg_val = builder.ins().fcvt_to_sint(types::I64, arg_val);
                                } else if actual_cl_ty == types::F64 && expected_cl_ty == types::I32
                                {
                                    // Float to i32 conversion
                                    arg_val = builder.ins().fcvt_to_sint(types::I32, arg_val);
                                }
                                // Other type mismatches are handled elsewhere or cause verifier errors
                            }
                        }
                        call_args.push(arg_val);
                    }

                    // Emit the call instruction
                    let call_mismatch = {
                        let decl = module.declarations().get_function_decl(cl_func_id);
                        let expected_params = decl.signature.params.len();
                        if expected_params != call_args.len() {
                            eprintln!("[CALL MISMATCH] In '{}': calling '{}' (MIR {:?}, CL {:?}): expected {} params, providing {} args, is_extern={}, env_added={}, sret={}",
                                function.name, called_func.name, func_id, cl_func_id,
                                expected_params, call_args.len(), is_extern_func, should_add_env, uses_sret);
                            for (pi, p) in called_func.signature.parameters.iter().enumerate() {
                                eprintln!("  MIR param[{}] '{}': {:?}", pi, p.name, p.ty);
                            }
                            true
                        } else {
                            false
                        }
                    };
                    // If call has wrong number of args, emit trap instead of crashing Cranelift.
                    // This handles imported functions with broken MIR that are never actually called.
                    if call_mismatch {
                        builder
                            .ins()
                            .trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
                        // Provide a dummy value for the dest register if needed
                        if let Some(dest_reg) = dest {
                            let dummy = builder.ins().iconst(types::I64, 0);
                            value_map.insert(*dest_reg, dummy);
                        }
                    } else {
                        let call_inst = builder.ins().call(func_ref, &call_args);

                        // Handle return value
                        // Special case: If the function returns Void but MIR has a dest register,
                        // just ignore the dest. This can happen with lambdas where the MIR was
                        // generated before the function signature was fully resolved.
                        if called_func.signature.return_type == crate::ir::IrType::Void {
                            // Void function - ignore dest register if present
                            // (MIR may have allocated one before signature was known)
                        } else if let Some(dest_reg) = dest {
                            if uses_sret {
                                // For sret, the "return value" is the pointer to the sret slot
                                value_map.insert(*dest_reg, sret_slot.unwrap());
                            } else {
                                // Normal return value
                                let results = builder.inst_results(call_inst);
                                if !results.is_empty() {
                                    let result_val = results[0];

                                    // Coerce return value to match expected MIR type
                                    // IMPORTANT: Only coerce if both the function signature return type AND
                                    // the MIR dest register type are primitive integers (not pointers/refs).
                                    // Truncating pointer values would cause runtime crashes.
                                    let actual_ret_ty = builder.func.dfg.value_type(result_val);

                                    // Check if the function signature says this is a primitive integer return
                                    let sig_return_is_primitive_int = matches!(
                                        &called_func.signature.return_type,
                                        crate::ir::IrType::I8
                                            | crate::ir::IrType::I16
                                            | crate::ir::IrType::I32
                                            | crate::ir::IrType::I64
                                            | crate::ir::IrType::U8
                                            | crate::ir::IrType::U16
                                            | crate::ir::IrType::U32
                                            | crate::ir::IrType::U64
                                            | crate::ir::IrType::Bool
                                    );

                                    // Check if MIR dest register type is also a primitive integer
                                    let mir_dest_is_primitive_int = function
                                        .register_types
                                        .get(dest_reg)
                                        .map(|ty| {
                                            matches!(
                                                ty,
                                                crate::ir::IrType::I8
                                                    | crate::ir::IrType::I16
                                                    | crate::ir::IrType::I32
                                                    | crate::ir::IrType::I64
                                                    | crate::ir::IrType::U8
                                                    | crate::ir::IrType::U16
                                                    | crate::ir::IrType::U32
                                                    | crate::ir::IrType::U64
                                                    | crate::ir::IrType::Bool
                                            )
                                        })
                                        .unwrap_or(false);

                                    let mir_expected_ty = function
                                        .register_types
                                        .get(dest_reg)
                                        .map(|ty| match ty {
                                            crate::ir::IrType::I8 => types::I8,
                                            crate::ir::IrType::I16 => types::I16,
                                            crate::ir::IrType::I32 => types::I32,
                                            crate::ir::IrType::I64 => types::I64,
                                            crate::ir::IrType::U8 => types::I8,
                                            crate::ir::IrType::U16 => types::I16,
                                            crate::ir::IrType::U32 => types::I32,
                                            crate::ir::IrType::U64 => types::I64,
                                            crate::ir::IrType::Bool => types::I8,
                                            _ => types::I64,
                                        })
                                        .unwrap_or(types::I64);

                                    // Only coerce if BOTH signature and MIR say it's a primitive int
                                    let final_val = if sig_return_is_primitive_int
                                        && mir_dest_is_primitive_int
                                        && actual_ret_ty != mir_expected_ty
                                        && actual_ret_ty.is_int()
                                        && mir_expected_ty.is_int()
                                    {
                                        debug!("Call return type coercion: actual={:?}, mir_expected={:?}, func={}",
                                    actual_ret_ty, mir_expected_ty, called_func.name);
                                        if actual_ret_ty.bits() > mir_expected_ty.bits() {
                                            // Truncate i64 -> i32
                                            builder.ins().ireduce(mir_expected_ty, result_val)
                                        } else {
                                            // Extend i32 -> i64
                                            builder.ins().sextend(mir_expected_ty, result_val)
                                        }
                                    } else {
                                        result_val
                                    };

                                    value_map.insert(*dest_reg, final_val);
                                } else {
                                    return Err(format!("Function call expected to return value but got none (func_id={:?}, dest={:?})", func_id, dest_reg));
                                }
                            }
                        }
                    } // end else (call_mismatch)
                }
            }

            IrInstruction::CallIndirect {
                dest,
                func_ptr,
                args,
                signature,
                arg_ownership: _,
                is_tail_call: _,
            } => {
                // Indirect function call (virtual call or closure call)
                // In our unified representation, func_ptr is ALWAYS a pointer to a Closure struct
                // { fn_ptr: i64, env_ptr: i64 }

                // Prepare arguments
                let mut call_args = Vec::new();
                for arg in args {
                    let arg_val = *value_map
                        .get(arg)
                        .ok_or_else(|| format!("Argument {:?} not found in value_map", arg))?;
                    call_args.push(arg_val);
                }

                // Get the closure object pointer
                let closure_ptr = *value_map.get(func_ptr).ok_or_else(|| {
                    format!("Function pointer {:?} not found in value_map", func_ptr)
                })?;

                // Load function pointer from offset 0
                let func_code_ptr = builder
                    .ins()
                    .load(types::I64, MemFlags::new(), closure_ptr, 0);

                // Load environment pointer from offset 8
                let env_ptr = builder
                    .ins()
                    .load(types::I64, MemFlags::new(), closure_ptr, 8);

                // Add environment pointer as first argument
                call_args.insert(0, env_ptr);

                // Determine function signature
                // We need to add the environment parameter to the signature
                let mut sig = module.make_signature();

                // Helper to add params to signature
                let add_params_to_sig = |sig: &mut Signature,
                                         param_types: &[IrType],
                                         ret_type: &IrType|
                 -> Result<(), String> {
                    // Check return type size to determine sret
                    let uses_sret = matches!(ret_type, IrType::Struct { .. });

                    if uses_sret {
                        sig.params
                            .push(AbiParam::special(types::I64, ArgumentPurpose::StructReturn));
                    }

                    // Add environment parameter
                    sig.params.push(AbiParam::new(types::I64));

                    // Add user parameters
                    for param_ty in param_types {
                        let cl_ty = Self::mir_type_to_cranelift_static(param_ty)?;
                        sig.params.push(AbiParam::new(cl_ty));
                    }
                    Ok(())
                };

                match signature {
                    IrType::Function {
                        params,
                        return_type,
                        varargs: _,
                    } => {
                        add_params_to_sig(&mut sig, params, return_type)?;

                        // Add return type
                        let cl_ret_ty = Self::mir_type_to_cranelift_static(return_type)?;
                        if cl_ret_ty != types::INVALID {
                            // Check for sret
                            if !matches!(return_type.as_ref(), IrType::Struct { .. }) {
                                sig.returns.push(AbiParam::new(cl_ret_ty));
                            }
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Invalid signature type for CallIndirect: {:?}",
                            signature
                        ))
                    }
                }

                let sig_ref = builder.import_signature(sig);

                // Emit the indirect call instruction
                let call_inst = builder
                    .ins()
                    .call_indirect(sig_ref, func_code_ptr, &call_args);
                let results = builder.inst_results(call_inst);

                // Map return value
                if let Some(dest_id) = dest {
                    if !results.is_empty() {
                        value_map.insert(*dest_id, results[0]);
                    }
                }
            }

            IrInstruction::MakeClosure {
                dest,
                func_id,
                captured_values,
            } => {
                // Create a closure object as a struct { fn_ptr: *u8, env_ptr: *u8 }
                //
                // Strategy:
                // 1. Allocate environment struct for captured values (if any)
                // 2. Allocate closure object struct (16 bytes: fn_ptr + env_ptr)
                // 3. Store function pointer and environment pointer into closure object
                // 4. Return pointer to closure object

                // Get the Cranelift FuncId for the lambda
                let cl_func_id = function_map.get(func_id).ok_or_else(|| {
                    format!("Lambda function {:?} not found in function_map", func_id)
                })?;

                // Import function and get its address
                let func_ref = module.declare_func_in_func(*cl_func_id, builder.func);
                let func_addr = builder.ins().func_addr(types::I64, func_ref);

                // Allocate environment for captured values (if any)
                let env_ptr = if !captured_values.is_empty() {
                    // Calculate environment size: 8 bytes per captured value
                    let env_size = (captured_values.len() * 8) as i64;

                    // Heap-allocate environment using malloc
                    // This is necessary because the closure may outlive the current stack frame
                    // (e.g., when passed to Thread.spawn())
                    let malloc_func_id = *runtime_functions
                        .get("malloc")
                        .ok_or_else(|| "malloc not found in runtime_functions".to_string())?;
                    let malloc_func_ref = module.declare_func_in_func(malloc_func_id, builder.func);

                    let size_arg = builder.ins().iconst(types::I64, env_size);
                    let inst = builder.ins().call(malloc_func_ref, &[size_arg]);
                    let env_addr = builder.inst_results(inst)[0];

                    // Store each captured value into the environment
                    for (i, captured_id) in captured_values.iter().enumerate() {
                        let captured_val = *value_map.get(captured_id).ok_or_else(|| {
                            format!("Captured value {:?} not found in value_map", captured_id)
                        })?;

                        // Calculate offset for this field (i * 8 bytes)
                        let offset = (i * 8) as i32;

                        // All environment slots are i64 (8 bytes) for uniformity
                        // If the value is smaller, extend it to i64
                        let val_type = builder.func.dfg.value_type(captured_val);
                        let value_to_store = match val_type {
                            types::I32 => {
                                // Sign-extend i32 to i64
                                builder.ins().sextend(types::I64, captured_val)
                            }
                            types::I8 => {
                                // Sign-extend i8 to i64
                                builder.ins().sextend(types::I64, captured_val)
                            }
                            types::I64 => {
                                // Already i64, use as-is
                                captured_val
                            }
                            _ => {
                                // For other types (pointers, floats, etc.), assume they're already pointer-sized
                                captured_val
                            }
                        };

                        // Store the i64 value at env_ptr + offset
                        builder
                            .ins()
                            .store(MemFlags::new(), value_to_store, env_addr, offset);
                    }

                    info!(
                        "Info: Allocated environment for {} captured variables",
                        captured_values.len()
                    );
                    env_addr
                } else {
                    // No captures - null environment pointer
                    builder.ins().iconst(types::I64, 0)
                };

                // Heap-allocate closure object struct: { fn_ptr: i64, env_ptr: i64 }
                // This is necessary because closures may outlive the current stack frame
                let malloc_func_id = *runtime_functions
                    .get("malloc")
                    .ok_or_else(|| "malloc not found in runtime_functions".to_string())?;
                let malloc_func_ref = module.declare_func_in_func(malloc_func_id, builder.func);

                let closure_size = builder.ins().iconst(types::I64, 16); // 2 pointers
                let inst = builder.ins().call(malloc_func_ref, &[closure_size]);
                let closure_obj_ptr = builder.inst_results(inst)[0];

                // Store function pointer at offset 0
                builder
                    .ins()
                    .store(MemFlags::new(), func_addr, closure_obj_ptr, 0);

                // Store environment pointer at offset 8
                builder
                    .ins()
                    .store(MemFlags::new(), env_ptr, closure_obj_ptr, 8);

                // Track the environment pointer for ClosureEnv instruction
                closure_environments.insert(*dest, env_ptr);

                // Return pointer to closure object struct
                value_map.insert(*dest, closure_obj_ptr);
            }

            IrInstruction::ClosureFunc { dest, closure } => {
                // Extract function pointer from closure
                // For now, closure is just the function pointer
                let closure_val = *value_map
                    .get(closure)
                    .ok_or_else(|| format!("Closure {:?} not found in value_map", closure))?;
                value_map.insert(*dest, closure_val);
            }

            IrInstruction::ClosureEnv { dest, closure: _ } => {
                // Extract environment pointer from the CURRENT function's environment parameter
                // The 'closure' argument to this instruction is usually the closure object itself,
                // but in our unified model, the environment is passed as a hidden parameter.
                // The MIR might still pass the closure object, but we just need the env param.

                if let Some(env_val) = current_env_param {
                    value_map.insert(*dest, env_val);
                } else {
                    // Should not happen if we set up current_env_param correctly
                    // But for safety, return null
                    let null_ptr = builder.ins().iconst(types::I64, 0);
                    value_map.insert(*dest, null_ptr);
                }
            }

            IrInstruction::Cast {
                dest,
                src,
                from_ty,
                to_ty,
            } => {
                // Type casting (e.g., int to float, float to int)
                let src_val = *value_map
                    .get(src)
                    .ok_or_else(|| {
                        let keys: Vec<_> = value_map.keys().collect();
                        format!("Cast source {:?} not found in value_map (function: {}, from_ty: {:?}, to_ty: {:?}, value_map keys: {:?})", src, function.name, from_ty, to_ty, keys)
                    })?;

                // Use the ACTUAL Cranelift value type, not the declared MIR from_ty
                // This handles cases where MIR type is wrong (e.g., Ptr(Void) for generic returns)
                let actual_src_ty = builder.func.dfg.value_type(src_val);
                let to_cl_ty = Self::mir_type_to_cranelift_static(to_ty)?;

                let result = match (actual_src_ty, to_cl_ty) {
                    // Same type - just copy (no conversion needed)
                    (from, to) if from == to => src_val,

                    // Int to Float conversions
                    (types::I32, types::F64) => builder.ins().fcvt_from_sint(types::F64, src_val),
                    (types::I64, types::F64) => builder.ins().fcvt_from_sint(types::F64, src_val),
                    (types::I32, types::F32) => builder.ins().fcvt_from_sint(types::F32, src_val),
                    (types::I64, types::F32) => builder.ins().fcvt_from_sint(types::F32, src_val),

                    // Float to Int conversions
                    (types::F64, types::I32) => builder.ins().fcvt_to_sint(types::I32, src_val),
                    (types::F64, types::I64) => builder.ins().fcvt_to_sint(types::I64, src_val),
                    (types::F32, types::I32) => builder.ins().fcvt_to_sint(types::I32, src_val),
                    (types::F32, types::I64) => builder.ins().fcvt_to_sint(types::I64, src_val),

                    // Float to Float conversions
                    (types::F32, types::F64) => builder.ins().fpromote(types::F64, src_val),
                    (types::F64, types::F32) => builder.ins().fdemote(types::F32, src_val),

                    // Int to Int conversions (sign extension or truncation)
                    (types::I32, types::I64) => builder.ins().sextend(types::I64, src_val),
                    (types::I64, types::I32) => builder.ins().ireduce(types::I32, src_val),

                    // Bool (I8) to integer conversions (zero-extend for raw value storage)
                    (types::I8, types::I32) => builder.ins().uextend(types::I32, src_val),
                    (types::I8, types::I64) => builder.ins().uextend(types::I64, src_val),
                    (types::I32, types::I8) => builder.ins().ireduce(types::I8, src_val),
                    (types::I64, types::I8) => builder.ins().ireduce(types::I8, src_val),

                    _ => {
                        return Err(format!(
                            "Unsupported cast from {:?} ({:?}) to {:?}",
                            actual_src_ty, from_ty, to_ty
                        ))
                    }
                };

                value_map.insert(*dest, result);
            }

            IrInstruction::BitCast { dest, src, ty } => {
                // BitCast - reinterpret bits without conversion
                // Used for Float <-> U64 raw value storage (preserves bit pattern)
                let src_val = *value_map
                    .get(src)
                    .ok_or_else(|| format!("BitCast source {:?} not found in value_map", src))?;

                let src_ty = builder.func.dfg.value_type(src_val);
                let dest_ty = Self::mir_type_to_cranelift_static(ty)?;

                let result = match (src_ty, dest_ty) {
                    // Float to Int (reinterpret bits)
                    (types::F64, types::I64) => {
                        builder.ins().bitcast(types::I64, MemFlags::new(), src_val)
                    }
                    (types::F32, types::I32) => {
                        builder.ins().bitcast(types::I32, MemFlags::new(), src_val)
                    }

                    // Int to Float (reinterpret bits)
                    (types::I64, types::F64) => {
                        builder.ins().bitcast(types::F64, MemFlags::new(), src_val)
                    }
                    (types::I32, types::F32) => {
                        builder.ins().bitcast(types::F32, MemFlags::new(), src_val)
                    }

                    // Int width conversions (sign-extend or truncate)
                    (types::I32, types::I64) => builder.ins().sextend(types::I64, src_val),
                    (types::I64, types::I32) => builder.ins().ireduce(types::I32, src_val),

                    // Same type - just copy
                    (from, to) if from == to => src_val,

                    _ => {
                        return Err(format!(
                            "Unsupported bitcast from {:?} to {:?}",
                            src_ty, dest_ty
                        ))
                    }
                };

                value_map.insert(*dest, result);
            }

            IrInstruction::GetElementPtr {
                dest,
                ptr,
                indices,
                ty,
            } => {
                // Get Element Pointer - compute address of field within struct
                // This is similar to LLVM's GEP instruction

                // debug!("Cranelift: GetElementPtr - ptr={:?}, indices={:?}, ty={:?}", ptr, indices, ty);

                let ptr_val = *value_map
                    .get(ptr)
                    .ok_or_else(|| format!("GEP ptr {:?} not found in value_map", ptr))?;

                // For now, we assume a single index (field index in struct)
                // More complex GEP operations (nested structs, arrays) need additional work
                if indices.len() != 1 {
                    return Err(format!(
                        "GEP with {} indices not yet supported (only single index supported)",
                        indices.len()
                    ));
                }

                let index_id = indices[0];
                let index_val = *value_map
                    .get(&index_id)
                    .ok_or_else(|| format!("GEP index {:?} not found in value_map", index_id))?;
                // Determine element size from the GEP type:
                // - Byte-pointer types (*u8, *i8): elem_size=1, index is already a byte offset
                // - Struct field types (*void, f64, i32, etc.): elem_size=8, all Rayzor
                //   object field slots are uniformly 8 bytes (class_alloc_sizes = field_index * 8)
                let elem_size: usize = match ty {
                    IrType::Ptr(inner) => match inner.as_ref() {
                        IrType::U8 | IrType::I8 => 1,
                        _ => 8,
                    },
                    _ => 8,
                };
                let size_val = builder.ins().iconst(types::I64, elem_size as i64);

                // Convert index to i64 if needed (only if not already i64)
                let index_ty = builder.func.dfg.value_type(index_val);
                let index_i64 = if index_ty == types::I64 {
                    index_val
                } else if index_ty.bits() < 64 {
                    builder.ins().sextend(types::I64, index_val)
                } else {
                    // Shouldn't happen, but handle gracefully
                    return Err(format!("GEP index has unsupported type {:?}", index_ty));
                };

                // Compute offset: index * elem_size
                let offset = builder.ins().imul(index_i64, size_val);

                // Add offset to base pointer
                let result_ptr = builder.ins().iadd(ptr_val, offset);

                // debug!("Cranelift: GEP result - dest={:?}", dest);
                value_map.insert(*dest, result_ptr);
            }

            IrInstruction::ExtractValue {
                dest,
                aggregate,
                indices,
            } => {
                // For struct field extraction, we need to calculate the offset and load
                // Get the aggregate value (should be a pointer to struct on stack)
                let aggregate_val = *value_map
                    .get(aggregate)
                    .ok_or_else(|| format!("Aggregate value {:?} not found", aggregate))?;

                // For now, handle simple single-index case (most common for structs)
                if indices.len() != 1 {
                    return Err(format!(
                        "ExtractValue with multiple indices not yet supported: {:?}",
                        indices
                    ));
                }

                let field_index = indices[0] as usize;

                // Get the struct type from the aggregate - check both parameters and locals
                // If not found, try to find the Load instruction that produced this value
                let aggregate_ty = function
                    .signature
                    .parameters
                    .iter()
                    .find(|p| p.reg == *aggregate)
                    .map(|p| &p.ty)
                    .or_else(|| function.locals.get(aggregate).map(|local| &local.ty))
                    .or_else(|| {
                        // Search for the Load instruction that produced this aggregate
                        for block in function.cfg.blocks.values() {
                            for inst in &block.instructions {
                                if let IrInstruction::Load { dest, ty, .. } = inst {
                                    if dest == aggregate {
                                        return Some(ty);
                                    }
                                }
                            }
                        }
                        None
                    })
                    .ok_or_else(|| format!("Type not found for aggregate {:?}", aggregate))?;

                // Calculate field offset based on struct layout
                let (field_offset, field_ty) = match aggregate_ty {
                    IrType::Struct { fields, .. } => {
                        if field_index >= fields.len() {
                            return Err(format!(
                                "Field index {} out of bounds for struct with {} fields",
                                field_index,
                                fields.len()
                            ));
                        }

                        // Calculate offset: sum of sizes of all previous fields
                        let offset: usize = fields
                            .iter()
                            .take(field_index)
                            .map(|f| CraneliftBackend::type_size(&f.ty))
                            .sum();

                        let field = &fields[field_index];
                        (offset, &field.ty)
                    }
                    _ => {
                        return Err(format!(
                            "ExtractValue on non-struct type: {:?}",
                            aggregate_ty
                        ));
                    }
                };

                // Add offset to base pointer
                let offset_val = builder.ins().iconst(types::I64, field_offset as i64);
                let field_ptr = builder.ins().iadd(aggregate_val, offset_val);

                // Load the field value
                let field_cl_ty = CraneliftBackend::mir_type_to_cranelift_static(field_ty)?;
                let field_value = builder
                    .ins()
                    .load(field_cl_ty, MemFlags::new(), field_ptr, 0);

                value_map.insert(*dest, field_value);
            }

            IrInstruction::FunctionRef { dest, func_id } => {
                // Get function reference as a pointer
                let cl_func_id = *function_map
                    .get(func_id)
                    .ok_or_else(|| format!("Function {:?} not found in function_map", func_id))?;

                // Import the function reference into the current function
                let func_ref = module.declare_func_in_func(cl_func_id, builder.func);

                // Convert function reference to an address (i64 pointer)
                let func_code_ptr = builder.ins().func_addr(types::I64, func_ref);

                // Create a Closure object { fn_ptr, env_ptr }
                // Even for static functions, we represent them as closures with null environment
                // This unifies the representation for CallIndirect

                // Allocate closure object (16 bytes)
                // We use malloc for consistency, though we could potentially optimize this
                let malloc_func_id = *runtime_functions
                    .get("malloc")
                    .ok_or_else(|| "malloc not found in runtime_functions".to_string())?;
                let malloc_func_ref = module.declare_func_in_func(malloc_func_id, builder.func);

                let closure_size = builder.ins().iconst(types::I64, 16); // 2 pointers
                let inst = builder.ins().call(malloc_func_ref, &[closure_size]);
                let closure_obj_ptr = builder.inst_results(inst)[0];

                // Store function pointer at offset 0
                builder
                    .ins()
                    .store(MemFlags::new(), func_code_ptr, closure_obj_ptr, 0);

                // Store null environment at offset 8
                let null_ptr = builder.ins().iconst(types::I64, 0);
                builder
                    .ins()
                    .store(MemFlags::new(), null_ptr, closure_obj_ptr, 8);

                value_map.insert(*dest, closure_obj_ptr);
            }

            IrInstruction::Undef { dest, ty } => {
                // Undefined value - use zero/null for simplicity
                let cl_ty = CraneliftBackend::mir_type_to_cranelift_static(ty)?;
                let undef_val = if cl_ty == types::INVALID {
                    // Void type - no value needed, but instruction expects one
                    // Use a dummy i64(0)
                    builder.ins().iconst(types::I64, 0)
                } else if cl_ty.is_int() {
                    builder.ins().iconst(cl_ty, 0)
                } else if cl_ty.is_float() {
                    if cl_ty == types::F32 {
                        builder.ins().f32const(0.0)
                    } else {
                        builder.ins().f64const(0.0)
                    }
                } else {
                    // Pointer or other type - use null (0)
                    builder.ins().iconst(types::I64, 0)
                };

                value_map.insert(*dest, undef_val);
            }

            IrInstruction::CreateStruct { dest, ty, fields } => {
                // Allocate stack space for the struct
                let struct_size = match ty {
                    IrType::Struct {
                        fields: field_tys, ..
                    } => field_tys
                        .iter()
                        .map(|f| CraneliftBackend::type_size(&f.ty))
                        .sum::<usize>(),
                    _ => return Err(format!("CreateStruct with non-struct type: {:?}", ty)),
                };

                // Create stack slot for struct
                let struct_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    struct_size as u32,
                    8, // 8-byte alignment
                ));

                let slot_addr = builder.ins().stack_addr(types::I64, struct_slot, 0);

                // Store each field at its offset
                if let IrType::Struct {
                    fields: field_tys, ..
                } = ty
                {
                    let mut offset = 0;
                    for (i, field_val_id) in fields.iter().enumerate() {
                        let field_val = *value_map
                            .get(field_val_id)
                            .ok_or_else(|| format!("Struct field {:?} not found", field_val_id))?;

                        builder
                            .ins()
                            .store(MemFlags::new(), field_val, slot_addr, offset as i32);

                        // Move offset forward by field size
                        offset += CraneliftBackend::type_size(&field_tys[i].ty);
                    }
                }

                // Return the stack address as the struct value
                value_map.insert(*dest, slot_addr);
            }

            IrInstruction::CreateUnion {
                dest,
                discriminant,
                value,
                ty: _,
            } => {
                // For now, represent union as a struct { tag: i32, value_ptr: i64 }
                // This is a simplified representation - proper implementation would use
                // tagged union with max variant size

                // Create tag value
                let tag_val = builder.ins().iconst(types::I32, *discriminant as i64);

                // Get the value (for now, just use the value as-is or convert to pointer)
                let value_val = *value_map
                    .get(value)
                    .ok_or_else(|| format!("Union value {:?} not found", value))?;

                // For simplicity, store tag and value separately in a struct-like layout
                // Allocate space for the union on stack
                let union_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    16, // tag (4 bytes) + value (8 bytes) + padding
                    8,  // 8-byte alignment
                ));

                let slot_addr = builder.ins().stack_addr(types::I64, union_slot, 0);

                // Store tag at offset 0
                builder.ins().store(MemFlags::new(), tag_val, slot_addr, 0);

                // Store value at offset 8 (after padding)
                let value_offset = 8i32;
                builder
                    .ins()
                    .store(MemFlags::new(), value_val, slot_addr, value_offset);

                // Return the stack address as the union value
                value_map.insert(*dest, slot_addr);
            }

            IrInstruction::PtrAdd {
                dest,
                ptr,
                offset,
                ty,
            } => {
                // Pointer arithmetic: ptr + offset
                // Get pointer value
                let ptr_val = *value_map
                    .get(ptr)
                    .ok_or_else(|| format!("PtrAdd ptr {:?} not found", ptr))?;

                // Get offset value
                let offset_val = *value_map
                    .get(offset)
                    .ok_or_else(|| format!("PtrAdd offset {:?} not found", offset))?;

                // Get the size of the pointee type
                let pointee_ty = match ty {
                    IrType::Ptr(inner) => inner.as_ref(),
                    _ => return Err(format!("PtrAdd on non-pointer type: {:?}", ty)),
                };
                let elem_size = CraneliftBackend::type_size(pointee_ty);

                // Calculate byte offset: offset * elem_size
                // Assume offset is already i64 (or convert if needed in the future)
                let size_val = builder.ins().iconst(types::I64, elem_size as i64);
                let byte_offset = builder.ins().imul(offset_val, size_val);

                // Add to pointer
                let result_ptr = builder.ins().iadd(ptr_val, byte_offset);
                value_map.insert(*dest, result_ptr);
            }

            IrInstruction::Free { ptr } => {
                // Free heap-allocated memory (Rust-style drop semantics)
                // Skip if ptr is not in value_map — this happens when the Free was emitted
                // in a branch but the ptr was defined in a different (sibling) branch.
                // At runtime, that branch's value never existed, so there's nothing to free.
                if let Some(&ptr_val) = value_map.get(ptr) {
                    // Get the libc free function
                    let free_func_id = *runtime_functions
                        .get("free")
                        .ok_or_else(|| "libc free not found in runtime_functions".to_string())?;
                    let free_func_ref = module.declare_func_in_func(free_func_id, builder.func);

                    // Call free(ptr) - returns void
                    builder.ins().call(free_func_ref, &[ptr_val]);

                    debug!("Free: Emitted call to free for {:?}", ptr);
                }
            }

            // === SIMD Vector Operations ===
            IrInstruction::VectorLoad { dest, ptr, vec_ty } => {
                let ptr_val = *value_map
                    .get(ptr)
                    .ok_or_else(|| format!("VectorLoad ptr {:?} not found", ptr))?;
                let cl_vec_ty = Self::mir_type_to_cranelift_static(vec_ty)?;
                // Load vector from memory (aligned load)
                let loaded = builder.ins().load(cl_vec_ty, MemFlags::new(), ptr_val, 0);
                value_map.insert(*dest, loaded);
            }

            IrInstruction::VectorStore {
                ptr,
                value,
                vec_ty: _,
            } => {
                let ptr_val = *value_map
                    .get(ptr)
                    .ok_or_else(|| format!("VectorStore ptr {:?} not found", ptr))?;
                let vec_val = *value_map
                    .get(value)
                    .ok_or_else(|| format!("VectorStore value {:?} not found", value))?;
                // Store vector to memory (aligned store)
                builder.ins().store(MemFlags::new(), vec_val, ptr_val, 0);
            }

            IrInstruction::VectorBinOp {
                dest,
                op,
                left,
                right,
                vec_ty,
            } => {
                let lhs = *value_map
                    .get(left)
                    .ok_or_else(|| format!("VectorBinOp left {:?} not found", left))?;
                let rhs = *value_map
                    .get(right)
                    .ok_or_else(|| format!("VectorBinOp right {:?} not found", right))?;

                // Get element type to determine if we use integer or float SIMD ops
                let is_float = match vec_ty {
                    IrType::Vector { element, .. } => {
                        matches!(**element, IrType::F32 | IrType::F64)
                    }
                    _ => false,
                };

                let result = match op {
                    crate::ir::BinaryOp::Add | crate::ir::BinaryOp::FAdd => {
                        if is_float {
                            builder.ins().fadd(lhs, rhs)
                        } else {
                            builder.ins().iadd(lhs, rhs)
                        }
                    }
                    crate::ir::BinaryOp::Sub | crate::ir::BinaryOp::FSub => {
                        if is_float {
                            builder.ins().fsub(lhs, rhs)
                        } else {
                            builder.ins().isub(lhs, rhs)
                        }
                    }
                    crate::ir::BinaryOp::Mul | crate::ir::BinaryOp::FMul => {
                        if is_float {
                            builder.ins().fmul(lhs, rhs)
                        } else {
                            builder.ins().imul(lhs, rhs)
                        }
                    }
                    crate::ir::BinaryOp::Div | crate::ir::BinaryOp::FDiv => {
                        if is_float {
                            builder.ins().fdiv(lhs, rhs)
                        } else {
                            // Integer division - use signed division
                            builder.ins().sdiv(lhs, rhs)
                        }
                    }
                    crate::ir::BinaryOp::And => builder.ins().band(lhs, rhs),
                    crate::ir::BinaryOp::Or => builder.ins().bor(lhs, rhs),
                    crate::ir::BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
                    _ => return Err(format!("Unsupported vector binary op: {:?}", op)),
                };
                value_map.insert(*dest, result);
            }

            IrInstruction::VectorSplat {
                dest,
                scalar,
                vec_ty,
            } => {
                let scalar_val = *value_map
                    .get(scalar)
                    .ok_or_else(|| format!("VectorSplat scalar {:?} not found", scalar))?;
                let cl_vec_ty = Self::mir_type_to_cranelift_static(vec_ty)?;
                // Splat scalar to all lanes
                let result = builder.ins().splat(cl_vec_ty, scalar_val);
                value_map.insert(*dest, result);
            }

            IrInstruction::VectorExtract {
                dest,
                vector,
                index,
            } => {
                let vec_val = *value_map
                    .get(vector)
                    .ok_or_else(|| format!("VectorExtract vector {:?} not found", vector))?;
                // Extract lane from vector
                let result = builder.ins().extractlane(vec_val, *index);
                value_map.insert(*dest, result);
            }

            IrInstruction::VectorInsert {
                dest,
                vector,
                scalar,
                index,
            } => {
                let vec_val = *value_map
                    .get(vector)
                    .ok_or_else(|| format!("VectorInsert vector {:?} not found", vector))?;
                let scalar_val = *value_map
                    .get(scalar)
                    .ok_or_else(|| format!("VectorInsert scalar {:?} not found", scalar))?;
                // Insert scalar into lane
                let result = builder.ins().insertlane(vec_val, scalar_val, *index);
                value_map.insert(*dest, result);
            }

            IrInstruction::VectorReduce { dest, op, vector } => {
                let vec_val = *value_map
                    .get(vector)
                    .ok_or_else(|| format!("VectorReduce vector {:?} not found", vector))?;

                // Get vector type info to determine lane count
                let vec_type = builder.func.dfg.value_type(vec_val);
                let lane_count = vec_type.lane_count() as u8;
                let is_float = vec_type.lane_type().is_float();

                // Implement horizontal reduction by extracting lanes and combining
                // Start with lane 0
                let mut result = builder.ins().extractlane(vec_val, 0);

                // Combine with remaining lanes
                for lane in 1..lane_count {
                    let lane_val = builder.ins().extractlane(vec_val, lane);
                    result = match op {
                        crate::ir::BinaryOp::Add | crate::ir::BinaryOp::FAdd => {
                            if is_float {
                                builder.ins().fadd(result, lane_val)
                            } else {
                                builder.ins().iadd(result, lane_val)
                            }
                        }
                        crate::ir::BinaryOp::Mul | crate::ir::BinaryOp::FMul => {
                            if is_float {
                                builder.ins().fmul(result, lane_val)
                            } else {
                                builder.ins().imul(result, lane_val)
                            }
                        }
                        crate::ir::BinaryOp::And => builder.ins().band(result, lane_val),
                        crate::ir::BinaryOp::Or => builder.ins().bor(result, lane_val),
                        crate::ir::BinaryOp::Xor => builder.ins().bxor(result, lane_val),
                        _ => return Err(format!("Unsupported vector reduce op: {:?}", op)),
                    };
                }
                value_map.insert(*dest, result);
            }

            IrInstruction::VectorUnaryOp {
                dest,
                op,
                operand,
                vec_ty: _,
            } => {
                let operand_val = *value_map
                    .get(operand)
                    .ok_or_else(|| format!("VectorUnaryOp operand {:?} not found", operand))?;

                let result = match op {
                    crate::ir::VectorUnaryOpKind::Sqrt => builder.ins().sqrt(operand_val),
                    crate::ir::VectorUnaryOpKind::Abs => builder.ins().fabs(operand_val),
                    crate::ir::VectorUnaryOpKind::Neg => builder.ins().fneg(operand_val),
                    crate::ir::VectorUnaryOpKind::Ceil => builder.ins().ceil(operand_val),
                    crate::ir::VectorUnaryOpKind::Floor => builder.ins().floor(operand_val),
                    crate::ir::VectorUnaryOpKind::Trunc => builder.ins().trunc(operand_val),
                    crate::ir::VectorUnaryOpKind::Round => builder.ins().nearest(operand_val),
                };
                value_map.insert(*dest, result);
            }

            IrInstruction::VectorMinMax {
                dest,
                op,
                left,
                right,
                vec_ty: _,
            } => {
                let lhs = *value_map
                    .get(left)
                    .ok_or_else(|| format!("VectorMinMax left {:?} not found", left))?;
                let rhs = *value_map
                    .get(right)
                    .ok_or_else(|| format!("VectorMinMax right {:?} not found", right))?;

                let result = match op {
                    crate::ir::VectorMinMaxKind::Min => builder.ins().fmin(lhs, rhs),
                    crate::ir::VectorMinMaxKind::Max => builder.ins().fmax(lhs, rhs),
                };
                value_map.insert(*dest, result);
            }

            // Global variable access - uses runtime functions for storage
            IrInstruction::LoadGlobal {
                dest,
                global_id,
                ty,
            } => {
                // Call rayzor_global_load(global_id) to get the stored value
                let global_id_val = builder
                    .ins()
                    .iconst(cranelift_codegen::ir::types::I64, global_id.0 as i64);

                // Get or declare the runtime function
                let load_func_id = if let Some(&func_id) =
                    runtime_functions.get("rayzor_global_load")
                {
                    func_id
                } else {
                    // Function not pre-declared, can't call it
                    tracing::warn!("[CRANELIFT] rayzor_global_load not found in runtime_functions");
                    let placeholder = builder.ins().iconst(cranelift_codegen::ir::types::I64, 0);
                    value_map.insert(*dest, placeholder);
                    return Ok(());
                };

                let load_func_ref = module.declare_func_in_func(load_func_id, builder.func);
                let call = builder.ins().call(load_func_ref, &[global_id_val]);
                let result = builder.inst_results(call)[0];

                // Cast result to the expected type if needed
                let cl_ty = match ty {
                    IrType::I32 => cranelift_codegen::ir::types::I32,
                    IrType::I64 => cranelift_codegen::ir::types::I64,
                    IrType::F32 => cranelift_codegen::ir::types::F32,
                    IrType::F64 => cranelift_codegen::ir::types::F64,
                    IrType::Bool => cranelift_codegen::ir::types::I8,
                    _ => cranelift_codegen::ir::types::I64,
                };

                let final_val = if cl_ty == cranelift_codegen::ir::types::I64 {
                    result
                } else if cl_ty.is_float() {
                    builder
                        .ins()
                        .bitcast(cl_ty, cranelift_codegen::ir::MemFlags::new(), result)
                } else {
                    builder.ins().ireduce(cl_ty, result)
                };

                tracing::debug!("[CRANELIFT] LoadGlobal {:?} - calling runtime", global_id);
                value_map.insert(*dest, final_val);
            }

            IrInstruction::StoreGlobal { global_id, value } => {
                // Call rayzor_global_store(global_id, value)
                let global_id_val = builder
                    .ins()
                    .iconst(cranelift_codegen::ir::types::I64, global_id.0 as i64);
                let val = *value_map
                    .get(value)
                    .ok_or_else(|| format!("StoreGlobal: value {:?} not found", value))?;

                // Extend value to i64 for storage
                let val_ty = builder.func.dfg.value_type(val);
                let val_i64 = if val_ty == cranelift_codegen::ir::types::I64 {
                    val
                } else if val_ty.is_float() {
                    builder.ins().bitcast(
                        cranelift_codegen::ir::types::I64,
                        cranelift_codegen::ir::MemFlags::new(),
                        val,
                    )
                } else {
                    builder
                        .ins()
                        .uextend(cranelift_codegen::ir::types::I64, val)
                };

                // Get or declare the runtime function
                if let Some(&store_func_id) = runtime_functions.get("rayzor_global_store") {
                    let store_func_ref = module.declare_func_in_func(store_func_id, builder.func);
                    builder
                        .ins()
                        .call(store_func_ref, &[global_id_val, val_i64]);
                    tracing::debug!("[CRANELIFT] StoreGlobal {:?} - calling runtime", global_id);
                } else {
                    tracing::warn!(
                        "[CRANELIFT] rayzor_global_store not found in runtime_functions"
                    );
                }
            }

            // TODO: Implement remaining instructions
            _ => {
                return Err(format!("Unsupported instruction: {:?}", instruction));
            }
        }

        Ok(())
    }

    /// Translate a terminator instruction (static method)
    fn translate_terminator_static(
        value_map: &HashMap<IrId, Value>,
        builder: &mut FunctionBuilder,
        terminator: &IrTerminator,
        block_map: &HashMap<IrBlockId, Block>,
        function: &IrFunction,
        sret_ptr: Option<Value>,
    ) -> Result<(), String> {
        use crate::ir::IrTerminator;

        match terminator {
            IrTerminator::Return { value } => {
                // debug!("Cranelift: Translating Return terminator, value={:?}", value);
                // debug!("Cranelift: value_map has {} entries", value_map.len());

                // If using sret, write the return value through the pointer and return void
                if let Some(sret) = sret_ptr {
                    if let Some(val_id) = value {
                        let val = *value_map
                            .get(val_id)
                            .ok_or_else(|| format!("Return value {:?} not found", val_id))?;

                        // Get the struct type to determine size
                        let struct_ty = function
                            .register_types
                            .get(val_id)
                            .or_else(|| function.locals.get(val_id).map(|l| &l.ty))
                            .ok_or_else(|| {
                                format!("Cannot find type for return value {:?}", val_id)
                            })?;

                        let struct_size = match struct_ty {
                            IrType::Struct { fields, .. } => fields
                                .iter()
                                .map(|f| CraneliftBackend::type_size(&f.ty))
                                .sum::<usize>(),
                            _ => return Err(format!("sret with non-struct type: {:?}", struct_ty)),
                        };

                        // Copy struct from source (val is a pointer to stack) to sret destination
                        // We need to do a memcpy-style copy of each field
                        if let IrType::Struct { fields, .. } = struct_ty {
                            let mut offset = 0;
                            for field in fields {
                                let field_ty =
                                    CraneliftBackend::mir_type_to_cranelift_static(&field.ty)?;
                                // Load from source struct
                                let field_val = builder.ins().load(
                                    field_ty,
                                    MemFlags::new(),
                                    val,
                                    offset as i32,
                                );
                                // Store to sret destination
                                builder.ins().store(
                                    MemFlags::new(),
                                    field_val,
                                    sret,
                                    offset as i32,
                                );
                                // Move offset forward
                                offset += CraneliftBackend::type_size(&field.ty);
                            }
                        }
                    }
                    // Return void for sret functions
                    builder.ins().return_(&[]);
                } else {
                    // Normal return path
                    if let Some(val_id) = value {
                        // debug!("Cranelift: Looking up return value {:?} in value_map", val_id);
                        let val = *value_map.get(val_id).ok_or_else(|| {
                            eprintln!("ERROR: Return value {:?} NOT FOUND in value_map!", val_id);
                            eprintln!(
                                "ERROR: Available values: {:?}",
                                value_map.keys().collect::<Vec<_>>()
                            );
                            format!("Return value {:?} not found", val_id)
                        })?;
                        // debug!("Cranelift: Found value, emitting return instruction");
                        builder.ins().return_(&[val]);
                    } else {
                        // Validate: void return should only happen for void functions
                        if function.signature.return_type != IrType::Void {
                            return Err(format!(
                                "Function '{}' has return type {:?} but encountered Return with no value. \
                                This usually indicates missing lowering of method calls in the function body.",
                                function.name, function.signature.return_type
                            ));
                        }
                        // debug!("Cranelift: Void return, no value");
                        builder.ins().return_(&[]);
                    }
                }
            }

            IrTerminator::Branch { target } => {
                let cl_block = *block_map
                    .get(target)
                    .ok_or_else(|| format!("Branch target {:?} not found", target))?;

                // Get current block to find phi node arguments
                let current_block_id = function
                    .cfg
                    .blocks
                    .iter()
                    .find(|(_, block)| std::ptr::eq(&block.terminator, terminator))
                    .map(|(id, _)| *id)
                    .ok_or_else(|| "Cannot find current block".to_string())?;

                // Collect phi node arguments for the target block (with type coercion if needed)
                let phi_args = Self::collect_phi_args_with_coercion(
                    value_map,
                    function,
                    *target,
                    current_block_id,
                    builder,
                )?;

                builder.ins().jump(cl_block, &phi_args);
            }

            IrTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let cond_val = *value_map
                    .get(condition)
                    .ok_or_else(|| format!("Condition value {:?} not found", condition))?;

                let true_block = *block_map
                    .get(true_target)
                    .ok_or_else(|| format!("True target {:?} not found", true_target))?;
                let false_block = *block_map
                    .get(false_target)
                    .ok_or_else(|| format!("False target {:?} not found", false_target))?;

                // Get current block to find phi node arguments
                let current_block_id = function
                    .cfg
                    .blocks
                    .iter()
                    .find(|(_, block)| std::ptr::eq(&block.terminator, terminator))
                    .map(|(id, _)| *id)
                    .ok_or_else(|| "Cannot find current block".to_string())?;

                // Collect phi node arguments for both targets (with type coercion if needed)
                let true_phi_args = Self::collect_phi_args_with_coercion(
                    value_map,
                    function,
                    *true_target,
                    current_block_id,
                    builder,
                )?;
                let false_phi_args = Self::collect_phi_args_with_coercion(
                    value_map,
                    function,
                    *false_target,
                    current_block_id,
                    builder,
                )?;

                builder.ins().brif(
                    cond_val,
                    true_block,
                    &true_phi_args,
                    false_block,
                    &false_phi_args,
                );
            }

            IrTerminator::Unreachable => {
                // Use a user trap code for unreachable (100 = unreachable)
                builder
                    .ins()
                    .trap(cranelift_codegen::ir::TrapCode::unwrap_user(100));
            }

            // TODO: Implement Switch and NoReturn
            _ => {
                return Err(format!("Unsupported terminator: {:?}", terminator));
            }
        }

        Ok(())
    }

    /// Translate a constant value to Cranelift IR (static method)
    fn translate_const_value(
        builder: &mut FunctionBuilder,
        value: &IrValue,
        function_map: &HashMap<IrFunctionId, FuncId>,
        runtime_functions: &mut HashMap<String, FuncId>,
        module: &mut JITModule,
        string_data: &mut HashMap<String, DataId>,
        string_counter: &mut usize,
    ) -> Result<Value, String> {
        use crate::ir::IrValue;

        let cl_value = match value {
            IrValue::I8(v) => builder.ins().iconst(types::I8, i64::from(*v)),
            IrValue::I16(v) => builder.ins().iconst(types::I16, i64::from(*v)),
            IrValue::I32(v) => {
                // For I32, need to handle negative values by treating as u32 first
                let as_u32 = *v as u32;
                builder.ins().iconst(types::I32, i64::from(as_u32))
            }
            IrValue::I64(v) => builder.ins().iconst(types::I64, *v),
            IrValue::U8(v) => builder.ins().iconst(types::I8, i64::from(*v)),
            IrValue::U16(v) => builder.ins().iconst(types::I16, i64::from(*v)),
            IrValue::U32(v) => builder.ins().iconst(types::I32, i64::from(*v)),
            IrValue::U64(v) => builder.ins().iconst(types::I64, *v as i64),
            IrValue::F32(v) => builder.ins().f32const(*v),
            IrValue::F64(v) => builder.ins().f64const(*v),
            IrValue::Bool(v) => builder.ins().iconst(types::I8, if *v { 1 } else { 0 }),
            IrValue::Null => builder.ins().iconst(types::I64, 0),
            IrValue::String(s) => {
                // Allocate string data in data section and call runtime to create HaxeString
                let data_id = if let Some(&existing) = string_data.get(s) {
                    existing
                } else {
                    // Create new data section entry for this string
                    let name = format!("str_{}", *string_counter);
                    *string_counter += 1;

                    let data_id = module
                        .declare_data(&name, Linkage::Local, false, false)
                        .map_err(|e| format!("Failed to declare string data: {}", e))?;

                    let mut data_desc = DataDescription::new();
                    data_desc.define(s.as_bytes().to_vec().into_boxed_slice());

                    module
                        .define_data(data_id, &data_desc)
                        .map_err(|e| format!("Failed to define string data: {}", e))?;

                    string_data.insert(s.clone(), data_id);
                    data_id
                };

                // Get pointer to the string data
                let gv = module.declare_data_in_func(data_id, builder.func);
                let str_ptr = builder.ins().global_value(types::I64, gv);
                let str_len = builder.ins().iconst(types::I64, s.len() as i64);

                // Get or declare haxe_string_literal runtime function
                let string_literal_func =
                    if let Some(&func_id) = runtime_functions.get("haxe_string_literal") {
                        func_id
                    } else {
                        // Declare haxe_string_literal(ptr: *const u8, len: usize) -> *mut HaxeString
                        let mut sig = module.make_signature();
                        sig.params.push(AbiParam::new(types::I64)); // ptr
                        sig.params.push(AbiParam::new(types::I64)); // len
                        sig.returns.push(AbiParam::new(types::I64)); // returns *mut HaxeString

                        let func_id = module
                            .declare_function("haxe_string_literal", Linkage::Import, &sig)
                            .map_err(|e| format!("Failed to declare haxe_string_literal: {}", e))?;

                        runtime_functions.insert("haxe_string_literal".to_string(), func_id);
                        func_id
                    };

                // Call haxe_string_literal(ptr, len) -> *mut HaxeString
                let func_ref = module.declare_func_in_func(string_literal_func, builder.func);
                let call = builder.ins().call(func_ref, &[str_ptr, str_len]);
                builder.inst_results(call)[0]
            }
            IrValue::Function(mir_func_id) => {
                // Get the Cranelift FuncId for this MIR function
                let cl_func_id = *function_map.get(mir_func_id).ok_or_else(|| {
                    format!("Function {:?} not found in function_map", mir_func_id)
                })?;

                // Import the function reference into the current function
                let func_ref = module.declare_func_in_func(cl_func_id, builder.func);

                // Convert function reference to an address (i64 pointer)
                builder.ins().func_addr(types::I64, func_ref)
            }
            _ => {
                return Err(format!("Unsupported constant value: {:?}", value));
            }
        };

        Ok(cl_value)
    }

    /// Convert MIR type to Cranelift type (static version for use without self)
    pub(super) fn mir_type_to_cranelift_static(ty: &IrType) -> Result<Type, String> {
        match ty {
            IrType::Void => Ok(types::INVALID), // Void functions have no return value
            IrType::I8 => Ok(types::I8),
            IrType::I16 => Ok(types::I16),
            IrType::I32 => Ok(types::I32),
            IrType::I64 => Ok(types::I64),
            IrType::U8 => Ok(types::I8),
            IrType::U16 => Ok(types::I16),
            IrType::U32 => Ok(types::I32),
            IrType::U64 => Ok(types::I64),
            IrType::F32 => Ok(types::F32),
            IrType::F64 => Ok(types::F64),
            IrType::Bool => Ok(types::I8),
            IrType::Ptr(_) => Ok(types::I64),
            IrType::Ref(_) => Ok(types::I64),
            IrType::Array(..) => Ok(types::I64),
            IrType::Slice(_) => Ok(types::I64),
            IrType::String => Ok(types::I64),
            IrType::Struct { .. } => Ok(types::I64),
            IrType::Union { .. } => Ok(types::I64),
            IrType::Any => Ok(types::I64),
            IrType::Function { .. } => Ok(types::I64),
            IrType::Opaque { .. } => Ok(types::I64),
            IrType::TypeVar(_) => Ok(types::I64), // Should be monomorphized
            IrType::Generic { .. } => Ok(types::I64), // Should be monomorphized
            // SIMD Vector types - 128-bit vectors for SSE/NEON compatibility
            IrType::Vector { element, count } => Self::mir_vector_to_cranelift(element, *count),
        }
    }

    /// Convert a MIR Vector type to the appropriate Cranelift SIMD type
    fn mir_vector_to_cranelift(element: &IrType, count: usize) -> Result<Type, String> {
        // Cranelift SIMD types are named as <element_type>x<count>
        // We support 128-bit vectors (SSE/NEON compatible)
        match (element, count) {
            // 128-bit float vectors
            (IrType::F32, 4) => Ok(types::F32X4), // 4x32 = 128 bits
            (IrType::F64, 2) => Ok(types::F64X2), // 2x64 = 128 bits

            // 128-bit integer vectors
            (IrType::I8 | IrType::U8, 16) => Ok(types::I8X16), // 16x8 = 128 bits
            (IrType::I16 | IrType::U16, 8) => Ok(types::I16X8), // 8x16 = 128 bits
            (IrType::I32 | IrType::U32, 4) => Ok(types::I32X4), // 4x32 = 128 bits
            (IrType::I64 | IrType::U64, 2) => Ok(types::I64X2), // 2x64 = 128 bits

            // 256-bit vectors (AVX) - future extension
            // (IrType::F32, 8) => Ok(types::F32X8),
            // (IrType::F64, 4) => Ok(types::F64X4),
            _ => Err(format!(
                "Unsupported vector type: {:?} x {}",
                element, count
            )),
        }
    }

    pub(super) fn mir_type_to_cranelift(&self, ty: &IrType) -> Result<Type, String> {
        Self::mir_type_to_cranelift_static(ty)
    }

    /// Get a pointer to the compiled function
    pub fn get_function_ptr(&mut self, mir_func_id: IrFunctionId) -> Result<*const u8, String> {
        let func_id = *self
            .function_map
            .get(&mir_func_id)
            .ok_or("Function not found")?;

        let code_ptr = self.module.get_finalized_function(func_id);

        Ok(code_ptr)
    }

    /// Call the main function (assuming it's void main() -> void)
    ///
    /// This function also waits for all spawned threads to complete before returning,
    /// ensuring that JIT code memory remains valid while threads are executing.
    /// Initialize all modules by registering enum RTTI from MIR metadata.
    /// Should be called before `call_main`.
    pub fn initialize_modules(
        &mut self,
        modules: &[std::sync::Arc<crate::ir::IrModule>],
    ) -> Result<(), String> {
        Self::register_enum_rtti_from_modules(modules);
        Self::register_class_rtti_from_modules(modules);
        Ok(())
    }

    /// Register class RTTI by walking MIR module type definitions directly.
    pub fn register_class_rtti_from_modules(modules: &[std::sync::Arc<crate::ir::IrModule>]) {
        use crate::ir::modules::IrTypeDefinition;
        use rayzor_runtime::type_system::register_class_from_mir;

        for module in modules {
            for (_id, typedef) in &module.types {
                if let IrTypeDefinition::Struct { fields, .. } = &typedef.definition {
                    let instance_fields: Vec<String> =
                        fields.iter().map(|f| f.name.clone()).collect();
                    let static_fields: Vec<String> = Vec::new();
                    let super_type_id = typedef.super_type_id.map(|t| t.0);

                    register_class_from_mir(
                        typedef.type_id.0,
                        &typedef.name,
                        super_type_id,
                        &instance_fields,
                        &static_fields,
                    );
                }
            }
        }
    }

    /// Register enum RTTI by walking MIR module type definitions directly.
    /// This avoids generating __init__ code that calls runtime functions via FFI.
    pub fn register_enum_rtti_from_modules(modules: &[std::sync::Arc<crate::ir::IrModule>]) {
        use crate::ir::modules::IrTypeDefinition;
        use rayzor_runtime::type_system::{register_enum_from_mir, ParamType};

        for module in modules {
            for (_id, typedef) in &module.types {
                if let IrTypeDefinition::Enum { variants, .. } = &typedef.definition {
                    let variant_data: Vec<(String, usize, Vec<ParamType>)> = variants
                        .iter()
                        .map(|v| {
                            let param_types: Vec<ParamType> = v
                                .fields
                                .iter()
                                .map(|f| Self::ir_type_to_param_type(&f.ty))
                                .collect();
                            (v.name.clone(), v.fields.len(), param_types)
                        })
                        .collect();

                    register_enum_from_mir(typedef.type_id.0, &typedef.name, &variant_data);
                    debug!(
                        "Registered enum RTTI '{}' (type_id={}) with {} variants",
                        typedef.name,
                        typedef.type_id.0,
                        variants.len()
                    );
                }
            }
        }
    }

    /// Map MIR IrType to runtime ParamType for RTTI registration.
    pub fn ir_type_to_param_type(ty: &IrType) -> rayzor_runtime::type_system::ParamType {
        use rayzor_runtime::type_system::ParamType;
        match ty {
            IrType::I32 | IrType::I64 => ParamType::Int,
            IrType::F32 | IrType::F64 => ParamType::Float,
            IrType::Bool => ParamType::Bool,
            IrType::String => ParamType::String,
            _ => ParamType::Dynamic,
        }
    }

    pub fn call_main(&mut self, module: &crate::ir::IrModule) -> Result<(), String> {
        // Call __vtable_init__ to register class virtual dispatch tables
        if let Some(vtable_init_func) = module
            .functions
            .values()
            .find(|f| f.name == "__vtable_init__")
        {
            if let Ok(vtable_init_ptr) = self.get_function_ptr(vtable_init_func.id) {
                debug!("  🔧 Calling __vtable_init__() for virtual dispatch tables...");
                unsafe {
                    let vtable_init_fn: extern "C" fn(i64) = std::mem::transmute(vtable_init_ptr);
                    vtable_init_fn(0); // null environment pointer
                }
            }
        }

        // Call __init__ function if it exists
        // __init__ handles module initialization like registering enum RTTI
        if let Some(init_func) = module.functions.values().find(|f| f.name == "__init__") {
            if let Ok(init_ptr) = self.get_function_ptr(init_func.id) {
                debug!("  🔧 Calling __init__() for module initialization...");
                unsafe {
                    let init_fn: extern "C" fn(i64) = std::mem::transmute(init_ptr);
                    init_fn(0); // null environment pointer
                }
            }
        }

        // Find the main function in the MIR module
        // Try various naming conventions: main, Main_main, Main.main, etc.
        let main_func = module
            .functions
            .values()
            .find(|f| {
                f.name == "main"
                    || f.name == "Main_main"
                    || f.name == "Main.main"
                    || f.name.ends_with("_main")
                    || f.name.ends_with(".main")
            })
            .ok_or_else(|| {
                // List available functions for debugging
                let func_names: Vec<_> = module
                    .functions
                    .values()
                    .filter(|f| !f.cfg.blocks.is_empty()) // Skip externs
                    .map(|f| &f.name)
                    .take(10)
                    .collect();
                format!(
                    "No main function found in module. Available functions (first 10): {:?}",
                    func_names
                )
            })?;

        // Get the function pointer
        let func_ptr = self.get_function_ptr(main_func.id)?;

        debug!("  🚀 Executing {}()...", main_func.name);

        // Call the main function (assuming it's void main() -> void)
        // This is unsafe because we're calling JIT-compiled code
        // NOTE: Cranelift adds a hidden environment parameter (i64) to non-extern Haxe
        // functions. We must pass a null pointer for this parameter.
        unsafe {
            let main_fn: extern "C" fn(i64) = std::mem::transmute(func_ptr);
            main_fn(0); // null environment pointer
        }

        // CRITICAL: Wait for all spawned threads to complete before returning
        // This prevents use-after-free when threads are still executing JIT code
        // and the JIT module is dropped
        debug!("  🔄 Waiting for spawned threads to complete...");
        rayzor_runtime::concurrency::rayzor_wait_all_threads();

        debug!("  ✅ Execution completed successfully!");

        Ok(())
    }
}

impl Default for CraneliftBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create Cranelift backend")
    }
}

impl CraneliftBackend {
    /// Get the size in bytes of an IR type
    fn type_size(ty: &crate::ir::IrType) -> usize {
        use crate::ir::IrType;
        match ty {
            IrType::I8 | IrType::U8 | IrType::Bool => 1,
            IrType::I16 | IrType::U16 => 2,
            IrType::I32 | IrType::U32 | IrType::F32 => 4,
            IrType::I64 | IrType::U64 | IrType::F64 => 8,
            IrType::Ptr(_) | IrType::Ref(_) => 8, // Assume 64-bit pointers
            IrType::Void => 0,
            IrType::Any => 8, // Boxed value pointer
            // SIMD vector types: element_size * count
            IrType::Vector { element, count } => Self::type_size(element) * count,
            _ => 8, // Default to pointer size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cranelift_backend_creation() {
        let backend = CraneliftBackend::new().unwrap();
        assert!(backend.function_map.is_empty());
    }
}
