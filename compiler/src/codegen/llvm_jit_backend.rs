//! # LLVM JIT Backend for Rayzor
//!
//! Implements Tier 3 (Maximum optimization) using LLVM's MCJIT for production-quality code generation.
//!
//! ## Architecture
//! - Uses LLVM 17.0 via inkwell bindings
//! - Translates Rayzor MIR (Mid-level IR) to LLVM IR
//! - Provides JIT compilation with aggressive optimization (-O3)
//! - Designed as drop-in replacement for Cranelift in hot paths
//!
//! ## Use Cases
//! 1. **Tier 3 in tiered JIT**: Optimize ultra-hot functions (>10k-100k calls)
//! 2. **AOT compilation**: Generate optimized native binaries
//! 3. **Profile-guided optimization**: Recompile based on runtime profiling
//!
//! ## Performance Target
//! - Compilation: 1-5s per function (slower than Cranelift)
//! - Runtime: 5-20x baseline (production C/C++ quality)
//! - Use only for truly hot code (<1% of functions)

#[cfg(feature = "llvm-backend")]
use inkwell::{
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    passes::PassBuilderOptions,
    targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetData, TargetMachine,
    },
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum},
    values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, GlobalValue, PhiValue,
        PointerValue,
    },
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
};

use crate::ir::{
    BinaryOp, CompareOp, IrBasicBlock, IrBlockId, IrFunction, IrFunctionId, IrGlobalId, IrId,
    IrInstruction, IrModule, IrPhiNode, IrTerminator, IrType, IrValue, UnaryOp, VectorMinMaxKind,
    VectorUnaryOpKind,
};
use std::collections::HashMap;
use std::sync::{Mutex, Once};

/// Static Once for thread-safe LLVM initialization
#[cfg(feature = "llvm-backend")]
static LLVM_INIT: Once = Once::new();

/// Global mutex to serialize all LLVM operations (LLVM is not fully thread-safe)
#[cfg(feature = "llvm-backend")]
static LLVM_MUTEX: Mutex<()> = Mutex::new(());

/// Global flag to track if LLVM compilation has been done in this process
///
/// IMPORTANT: LLVM does not handle multiple contexts being created and leaked
/// in the same process well - it leads to memory corruption and segfaults.
/// This flag ensures only ONE LLVM compilation happens per process.
#[cfg(feature = "llvm-backend")]
static LLVM_GLOBAL_COMPILED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Global storage for LLVM-compiled function pointers
///
/// When multiple backends exist in the same process, only ONE can do LLVM compilation.
/// This global map stores the compiled pointers so other backends can reuse them.
/// Key is function name (stable across backends), value is function pointer.
/// Uses Mutex<Option<>> instead of OnceLock to allow reset for benchmarking.
#[cfg(feature = "llvm-backend")]
static LLVM_GLOBAL_POINTERS: std::sync::Mutex<Option<HashMap<String, usize>>> =
    std::sync::Mutex::new(None);

/// Check if LLVM compilation has already been done globally
#[cfg(feature = "llvm-backend")]
pub fn is_llvm_compiled_globally() -> bool {
    LLVM_GLOBAL_COMPILED.load(std::sync::atomic::Ordering::Acquire)
}

/// Mark LLVM compilation as done globally and store the function pointers
#[cfg(feature = "llvm-backend")]
pub fn mark_llvm_compiled_globally_with_pointers(pointers: HashMap<String, usize>) {
    *LLVM_GLOBAL_POINTERS.lock().unwrap() = Some(pointers);
    LLVM_GLOBAL_COMPILED.store(true, std::sync::atomic::Ordering::Release);
}

/// Get the globally stored LLVM function pointers (if available)
/// Returns a clone to avoid holding the lock
#[cfg(feature = "llvm-backend")]
pub fn get_global_llvm_pointers() -> Option<HashMap<String, usize>> {
    LLVM_GLOBAL_POINTERS.lock().unwrap().clone()
}

/// Reset LLVM global state for benchmarking
///
/// This allows each benchmark target to do its own fresh LLVM compilation
/// instead of reusing pointers from a previous target's compilation.
/// IMPORTANT: Only use this for benchmarking - in production, reusing
/// compiled code is more efficient.
#[cfg(feature = "llvm-backend")]
pub fn reset_llvm_global_state() {
    LLVM_GLOBAL_COMPILED.store(false, std::sync::atomic::Ordering::Release);
    *LLVM_GLOBAL_POINTERS.lock().unwrap() = None;
}

/// Mark LLVM compilation as done globally (legacy, without pointers)
#[cfg(feature = "llvm-backend")]
pub fn mark_llvm_compiled_globally() {
    LLVM_GLOBAL_COMPILED.store(true, std::sync::atomic::Ordering::Release);
}

/// Initialize LLVM once (thread-safe)
///
/// IMPORTANT: Call this from the main thread before spawning any background threads
/// that will use LLVM. This ensures LLVM's global state is initialized safely.
#[cfg(feature = "llvm-backend")]
pub fn init_llvm_once() {
    LLVM_INIT.call_once(|| {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize LLVM native target");
        ExecutionEngine::link_in_mc_jit();
    });
}

/// Acquire the global LLVM lock - must be held during all LLVM operations
#[cfg(feature = "llvm-backend")]
pub fn llvm_lock() -> std::sync::MutexGuard<'static, ()> {
    LLVM_MUTEX.lock().unwrap()
}

/// LLVM JIT backend using MCJIT
///
/// Compiles Rayzor MIR to native code using LLVM's aggressive optimizations.
/// Used as Tier 3 in the tiered compilation system for ultra-hot functions.
#[cfg(feature = "llvm-backend")]
pub struct LLVMJitBackend<'ctx> {
    /// LLVM context (lifetime-bound)
    context: &'ctx Context,

    /// LLVM module
    module: Module<'ctx>,

    /// LLVM IR builder
    builder: Builder<'ctx>,

    /// JIT execution engine
    execution_engine: Option<ExecutionEngine<'ctx>>,

    /// Maps MIR value IDs to LLVM values
    value_map: HashMap<IrId, BasicValueEnum<'ctx>>,

    /// Maps MIR function IDs to LLVM functions
    function_map: HashMap<IrFunctionId, FunctionValue<'ctx>>,

    /// Maps MIR block IDs to LLVM basic blocks
    block_map: HashMap<IrBlockId, BasicBlock<'ctx>>,

    /// Maps phi node destination IDs to LLVM phi instructions
    phi_map: HashMap<IrId, PhiValue<'ctx>>,

    /// Function pointers cache (usize for thread safety)
    function_pointers: HashMap<IrFunctionId, usize>,

    /// Optimization level
    opt_level: OptimizationLevel,

    /// Target data for architecture-specific type sizes/alignment
    target_data: Option<TargetData>,

    /// Runtime symbols (name -> pointer) for FFI calls
    runtime_symbols: HashMap<String, usize>,

    /// Extern function IDs (no hidden env parameter)
    extern_function_ids: std::collections::HashSet<IrFunctionId>,

    /// Functions that use sret (struct return via hidden pointer parameter)
    sret_function_ids: std::collections::HashSet<IrFunctionId>,

    /// Current sret pointer for the function being compiled
    /// Set at the start of compile_function_body, used in Return terminator
    current_sret_ptr: Option<inkwell::values::PointerValue<'ctx>>,

    /// AOT mode: when true, struct→ptr coercion for extern calls uses
    /// alloca+store (C ABI), and per-function verification prints errors.
    aot_mode: bool,

    /// IrIds that were allocated via alloca (stack). Free instructions targeting
    /// these are no-ops. All other Free instructions call libc free().
    alloca_ids: std::collections::HashSet<IrId>,

    /// LLVM global variables for Haxe static fields (inline access, no FFI)
    /// Vec indexed by IrGlobalId.0 for O(1) lookup (no hashing overhead)
    global_vars: Vec<Option<GlobalValue<'ctx>>>,
}

#[cfg(feature = "llvm-backend")]
impl<'ctx> LLVMJitBackend<'ctx> {
    /// Create a new LLVM JIT backend with aggressive optimization (Tier 3)
    pub fn new(context: &'ctx Context) -> Result<Self, String> {
        Self::with_opt_level(context, OptimizationLevel::Aggressive)
    }

    /// Create with custom optimization level
    pub fn with_opt_level(
        context: &'ctx Context,
        opt_level: OptimizationLevel,
    ) -> Result<Self, String> {
        // Initialize LLVM once (thread-safe)
        init_llvm_once();

        // Create module
        let module = context.create_module("rayzor_jit");
        let builder = context.create_builder();

        // Get target data for the native target
        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple)
            .map_err(|e| format!("Failed to get target from triple: {}", e))?;
        let target_machine = target
            .create_target_machine(
                &target_triple,
                TargetMachine::get_host_cpu_name()
                    .to_str()
                    .unwrap_or("generic"),
                TargetMachine::get_host_cpu_features()
                    .to_str()
                    .unwrap_or(""),
                opt_level,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or("Failed to create target machine")?;

        let target_data = target_machine.get_target_data();

        // Set the module's target triple and data layout to match the host.
        // Without this, MCJIT's internal TargetMachine may use generic defaults
        // instead of the host CPU's features (FMA, AVX2, etc.), causing
        // significantly worse codegen on x86_64 Linux.
        module.set_triple(&target_triple);
        module.set_data_layout(&target_data.get_data_layout());

        Ok(Self {
            context,
            module,
            builder,
            execution_engine: None,
            value_map: HashMap::new(),
            function_map: HashMap::new(),
            block_map: HashMap::new(),
            phi_map: HashMap::new(),
            function_pointers: HashMap::new(),
            opt_level,
            target_data: Some(target_data),
            runtime_symbols: HashMap::new(),
            extern_function_ids: std::collections::HashSet::new(),
            sret_function_ids: std::collections::HashSet::new(),
            current_sret_ptr: None,
            aot_mode: false,
            alloca_ids: std::collections::HashSet::new(),
            global_vars: Vec::new(),
        })
    }

    /// Create a backend in AOT mode. In AOT mode, struct→ptr coercion for
    /// extern (C ABI) calls uses alloca+store instead of field extraction.
    pub fn with_aot_mode(
        context: &'ctx Context,
        opt_level: OptimizationLevel,
    ) -> Result<Self, String> {
        let mut backend = Self::with_opt_level(context, opt_level)?;
        backend.aot_mode = true;
        Ok(backend)
    }

    /// Create a new LLVM JIT backend with runtime symbols for FFI
    pub fn with_symbols(
        context: &'ctx Context,
        symbols: &[(&str, *const u8)],
    ) -> Result<Self, String> {
        // Check for environment variable to control optimization level
        let opt_level = match std::env::var("RAYZOR_LLVM_OPT").ok().as_deref() {
            Some("0") => OptimizationLevel::None,
            Some("1") => OptimizationLevel::Less,
            Some("2") => OptimizationLevel::Default,
            _ => OptimizationLevel::Aggressive, // Default to O3
        };
        let mut backend = Self::with_opt_level(context, opt_level)?;
        for (name, ptr) in symbols {
            backend
                .runtime_symbols
                .insert(name.to_string(), *ptr as usize);
        }
        Ok(backend)
    }

    /// Get the size of a type in bytes according to the target architecture
    pub fn get_type_size(&self, ty: &IrType) -> Result<u64, String> {
        let llvm_ty = self.translate_type(ty)?;
        if let Some(ref target_data) = self.target_data {
            Ok(target_data.get_store_size(&llvm_ty))
        } else {
            Err("Target data not available".to_string())
        }
    }

    /// Get the alignment of a type in bytes according to the target architecture
    pub fn get_type_alignment(&self, ty: &IrType) -> Result<u32, String> {
        let llvm_ty = self.translate_type(ty)?;
        if let Some(ref target_data) = self.target_data {
            Ok(target_data.get_abi_alignment(&llvm_ty))
        } else {
            Err("Target data not available".to_string())
        }
    }

    /// Get pointer size in bytes for the target architecture
    pub fn get_pointer_size(&self) -> u32 {
        if let Some(ref target_data) = self.target_data {
            target_data.get_pointer_byte_size(None)
        } else {
            8 // Default to 64-bit
        }
    }

    /// Fast-math flags bitmask for floating-point optimizations.
    /// LLVM FastMathFlags: AllowReassoc(1) | NoNaNs(2) | NoInfs(4) | NoSignedZeros(8) |
    ///                     AllowReciprocal(16) | AllowContract(32) | ApproxFunc(64)
    ///
    /// Using conservative flags WITHOUT AllowContract:
    /// - NoNaNs(2) + NoInfs(4) + NoSignedZeros(8) = 14
    /// - AllowContract is DISABLED because LLVM's O3 pass manager will perform
    ///   cross-block FMA fusion when it's enabled, changing FP results in
    ///   iteration-heavy loops (e.g. mandelbrot checksum 112798515 → 112798531).
    ///   Same-block FMA is handled explicitly via llvm.fma.f64 intrinsics in
    ///   try_extract_fmul_llvm(), which doesn't require AllowContract.
    /// - AllowReassoc is DISABLED as it can change results due to FP non-associativity
    /// - ApproxFunc is DISABLED as it uses approximations for math functions
    /// - AllowReciprocal is DISABLED as it can change division precision
    const FAST_MATH_FLAGS: u32 = 0x0E; // NoNaNs + NoInfs + NoSignedZeros (14)

    /// Create tuned `PassBuilderOptions` for LLVM optimization passes.
    ///
    /// Explicitly enables loop vectorization, unrolling, interleaving, and SLP
    /// vectorization. While `default<O3>` enables many of these, explicit flags
    /// ensure they are active on all platforms — especially x86_64 Linux where
    /// the defaults may be more conservative than on aarch64 macOS.
    fn create_pass_options() -> PassBuilderOptions {
        let opts = PassBuilderOptions::create();
        opts.set_loop_vectorization(true);
        opts.set_loop_unrolling(true);
        opts.set_loop_interleaving(true);
        opts.set_loop_slp_vectorization(true);
        opts.set_merge_functions(true);
        opts
    }

    /// Apply fast-math flags to a float instruction for aggressive optimization.
    /// This enables LLVM to perform optimizations like:
    /// - Reassociation of floating-point operations
    /// - Use of reciprocals instead of division
    /// - Contraction of multiply-add sequences into FMA
    /// - Approximation of transcendental functions
    #[inline]
    fn apply_fast_math(&self, result: inkwell::values::FloatValue<'ctx>) {
        // Get the instruction that produced this value and set fast-math flags
        // This is safe to call on any FloatValue - returns None if not an instruction
        if let Some(inst) = result.as_instruction_value() {
            inst.set_fast_math_flags(Self::FAST_MATH_FLAGS);
        }
    }

    /// Check if an LLVM float value was produced by an fmul in the same basic block.
    /// Returns the two fmul operands if so, enabling FMA fusion.
    /// Only fuses same-block to match Cranelift backend behavior and hxcpp results.
    fn try_extract_fmul_llvm(
        &self,
        value: inkwell::values::FloatValue<'ctx>,
    ) -> Option<(
        inkwell::values::FloatValue<'ctx>,
        inkwell::values::FloatValue<'ctx>,
    )> {
        if std::env::var("RAYZOR_NO_FMA").is_ok() {
            return None;
        }
        let inst = value.as_instruction_value()?;
        if inst.get_opcode() != inkwell::values::InstructionOpcode::FMul {
            return None;
        }
        // Same-block check
        let inst_block = inst.get_parent()?;
        let current_block = self.builder.get_insert_block()?;
        if inst_block != current_block {
            return None;
        }
        let a = inst.get_operand(0)?.left()?.into_float_value();
        let b = inst.get_operand(1)?.left()?.into_float_value();
        Some((a, b))
    }

    /// Build an FMA intrinsic call: fma(a, b, c) = a * b + c
    fn build_fma(
        &self,
        a: inkwell::values::FloatValue<'ctx>,
        b: inkwell::values::FloatValue<'ctx>,
        c: inkwell::values::FloatValue<'ctx>,
        name: &str,
    ) -> Result<inkwell::values::FloatValue<'ctx>, String> {
        use inkwell::intrinsics::Intrinsic;
        let fma_intrinsic =
            Intrinsic::find("llvm.fma.f64").ok_or("llvm.fma.f64 intrinsic not found")?;
        let f64_type = self.context.f64_type();
        let fma_func = fma_intrinsic
            .get_declaration(&self.module, &[f64_type.into()])
            .ok_or("Failed to get llvm.fma.f64 declaration")?;
        let result = self
            .builder
            .build_call(fma_func, &[a.into(), b.into(), c.into()], name)
            .map_err(|e| format!("Failed to build fma call: {}", e))?
            .try_as_basic_value()
            .left()
            .ok_or("fma intrinsic returned void")?
            .into_float_value();
        self.apply_fast_math(result);
        Ok(result)
    }

    /// Compile a single function (for tiered JIT)
    ///
    /// This is the main entry point for Tier 3 optimization.
    /// Compiles one function at maximum optimization level.
    pub fn compile_single_function(
        &mut self,
        func_id: IrFunctionId,
        function: &IrFunction,
    ) -> Result<(), String> {
        // Declare the function
        let llvm_func = self.declare_function(func_id, function)?;

        // Compile the function body
        self.compile_function_body(func_id, function, llvm_func)?;

        // Create execution engine if not exists
        if self.execution_engine.is_none() {
            let engine = self
                .module
                .create_jit_execution_engine(self.opt_level)
                .map_err(|e| format!("Failed to create JIT execution engine: {}", e))?;
            self.execution_engine = Some(engine);
        }

        // Get function pointer using the mangled name
        let func_name = Self::mangle_function_name(&function.name);
        if let Some(ref engine) = self.execution_engine {
            let fn_ptr = engine.get_function_address(&func_name).map_err(|e| {
                format!(
                    "Failed to get function address for '{}': {:?}",
                    func_name, e
                )
            })?;

            self.function_pointers.insert(func_id, fn_ptr as usize);
        }

        Ok(())
    }

    /// Get a compiled function pointer
    pub fn get_function_ptr(&mut self, func_id: IrFunctionId) -> Result<*const u8, String> {
        // Check cache first
        if let Some(&addr) = self.function_pointers.get(&func_id) {
            return Ok(addr as *const u8);
        }

        // JIT-compile on demand
        let llvm_func = self
            .function_map
            .get(&func_id)
            .ok_or_else(|| format!("Function {:?} not found in function_map", func_id))?;
        let func_name = llvm_func.get_name().to_string_lossy().to_string();

        let engine = self
            .execution_engine
            .as_ref()
            .ok_or("Execution engine not initialized - call finalize() first")?;

        let fn_ptr = engine
            .get_function_address(&func_name)
            .map_err(|e| format!("Failed to get function address for {}: {}", func_name, e))?;

        // Cache for future calls
        self.function_pointers.insert(func_id, fn_ptr as usize);

        Ok(fn_ptr as *const u8)
    }

    /// Get all compiled function pointers
    /// Call this after finalize() to get all available function addresses
    /// Get a single function pointer by ID (lazy resolution, avoids bulk compilation)
    pub fn get_function_pointer_by_id(&mut self, func_id: IrFunctionId) -> Option<usize> {
        if let Some(&ptr) = self.function_pointers.get(&func_id) {
            return Some(ptr);
        }
        let engine = self.execution_engine.as_ref()?;
        let llvm_func = self.function_map.get(&func_id)?;
        let func_name = llvm_func.get_name().to_string_lossy().to_string();
        if let Ok(fn_ptr) = engine.get_function_address(&func_name) {
            if fn_ptr != 0 {
                self.function_pointers.insert(func_id, fn_ptr as usize);
                return Some(fn_ptr as usize);
            }
        }
        None
    }

    pub fn get_all_function_pointers(&mut self) -> Result<HashMap<IrFunctionId, usize>, String> {
        let engine = self
            .execution_engine
            .as_ref()
            .ok_or("Execution engine not initialized - call finalize() first")?;

        // Get pointers for all functions in function_map that we haven't cached yet
        let mut null_ptrs: Vec<String> = Vec::new();
        for (&func_id, llvm_func) in &self.function_map {
            if !self.function_pointers.contains_key(&func_id) {
                let func_name = llvm_func.get_name().to_string_lossy().to_string();
                if let Ok(fn_ptr) = engine.get_function_address(&func_name) {
                    if fn_ptr != 0 {
                        self.function_pointers.insert(func_id, fn_ptr as usize);
                    } else {
                        null_ptrs.push(format!("{} ({:?})", func_name, func_id));
                    }
                } else {
                    null_ptrs.push(format!("{} ({:?}) [err]", func_name, func_id));
                }
            }
        }
        if !null_ptrs.is_empty() {
            tracing::warn!(
                "[LLVM] {} functions returned null/error pointers: {:?}",
                null_ptrs.len(),
                null_ptrs
            );
        }

        Ok(self.function_pointers.clone())
    }

    /// Declare all functions in a module without compiling their bodies
    ///
    /// Call this for ALL modules first before calling compile_module_bodies.
    /// This ensures all function references can be resolved across modules.
    pub fn declare_module(&mut self, module: &IrModule) -> Result<(), String> {
        // Pre-allocate global_vars Vec for O(1) access without bounds checking
        let max_global_id = module.next_global_id as usize;
        if max_global_id > self.global_vars.len() {
            self.global_vars.resize(max_global_id, None);
        }

        // IMPORTANT: Declare extern functions FIRST so they get the original names
        // for linking with runtime symbols. Regular functions will get unique names
        // if there's a conflict.
        for (func_id, extern_fn) in &module.extern_functions {
            self.declare_extern_function(*func_id, extern_fn)?;
        }
        // Declare regular functions (with hidden env parameter)
        for (func_id, function) in &module.functions {
            self.declare_function(*func_id, function)?;
        }
        Ok(())
    }

    /// Declare an external function (FFI/runtime function with no body)
    fn declare_extern_function(
        &mut self,
        func_id: IrFunctionId,
        extern_fn: &crate::ir::IrExternFunction,
    ) -> Result<FunctionValue<'ctx>, String> {
        // Track this as an extern function (no hidden env parameter)
        self.extern_function_ids.insert(func_id);

        // Translate parameter types (NO env param for extern C functions)
        let param_types: Result<Vec<BasicMetadataTypeEnum>, _> = extern_fn
            .signature
            .parameters
            .iter()
            .filter(|param| param.ty != IrType::Void)
            .map(|param| self.translate_type(&param.ty).map(|t| t.into()))
            .collect();
        let param_types = param_types?;

        // Translate return type
        let fn_type = if extern_fn.signature.return_type == IrType::Void {
            self.context.void_type().fn_type(&param_types, false)
        } else {
            let return_type = self.translate_type(&extern_fn.signature.return_type)?;
            return_type.fn_type(&param_types, false)
        };

        let func_name = Self::mangle_function_name(&extern_fn.name);

        // Replace known math runtime functions with LLVM intrinsic wrappers
        // so LLVM can inline them (e.g. fsqrt instruction instead of function call).
        // This is critical for performance in math-heavy code like nbody/mandelbrot.
        // Enabled for both JIT and AOT modes.
        if let Some(llvm_func) = self.try_create_math_intrinsic(&func_name, &fn_type)? {
            self.function_map.insert(func_id, llvm_func);
            return Ok(llvm_func);
        }

        // Replace Std functions with inline implementations (e.g., Std.int → fptosi)
        if let Some(llvm_func) = self.try_create_std_intrinsic(&func_name, &fn_type)? {
            self.function_map.insert(func_id, llvm_func);
            return Ok(llvm_func);
        }

        // Replace array operations with inline implementations
        // HaxeArray layout: { ptr, len, cap, elem_size } - all 8 bytes each
        if let Some(llvm_func) = self.try_create_array_intrinsic(&func_name, &fn_type)? {
            self.function_map.insert(func_id, llvm_func);
            return Ok(llvm_func);
        }

        // Check if already declared with MATCHING signature
        if let Some(existing_func) = self.module.get_function(&func_name) {
            let existing_params = existing_func.get_type().get_param_types();
            // Only reuse if signature matches (same number of params)
            if existing_params.len() == param_types.len() {
                self.function_map.insert(func_id, existing_func);
                return Ok(existing_func);
            }
            // Signature mismatch - create with unique name to avoid conflict
            let unique_name = format!("{}__extern_{}", func_name, func_id.0);
            let llvm_func = self.module.add_function(
                &unique_name,
                fn_type,
                Some(inkwell::module::Linkage::External),
            );
            self.function_map.insert(func_id, llvm_func);
            return Ok(llvm_func);
        }

        // Add function with external linkage
        let llvm_func = self.module.add_function(
            &func_name,
            fn_type,
            Some(inkwell::module::Linkage::External),
        );

        self.function_map.insert(func_id, llvm_func);
        Ok(llvm_func)
    }

    /// Replace known math runtime functions with inline LLVM intrinsic wrappers.
    /// Returns Some(func) if replaced, None if not a known math function.
    /// Used for both JIT and AOT modes to convert function calls to native instructions.
    fn try_create_math_intrinsic(
        &self,
        func_name: &str,
        fn_type: &inkwell::types::FunctionType<'ctx>,
    ) -> Result<Option<FunctionValue<'ctx>>, String> {
        use inkwell::intrinsics::Intrinsic;

        // Map runtime function names to LLVM intrinsic names
        let intrinsic_name = match func_name {
            "haxe_math_sqrt" => "llvm.sqrt.f64",
            "haxe_math_abs" => "llvm.fabs.f64",
            "haxe_math_floor" => "llvm.floor.f64",
            "haxe_math_ceil" => "llvm.ceil.f64",
            "haxe_math_round" => "llvm.round.f64",
            "haxe_math_sin" => "llvm.sin.f64",
            "haxe_math_cos" => "llvm.cos.f64",
            "haxe_math_exp" => "llvm.exp.f64",
            "haxe_math_log" => "llvm.log.f64",
            "haxe_math_pow" => "llvm.pow.f64",
            "haxe_math_fround" => "llvm.round.f64",
            _ => return Ok(None),
        };

        let intrinsic = Intrinsic::find(intrinsic_name)
            .ok_or_else(|| format!("LLVM intrinsic {} not found", intrinsic_name))?;

        let f64_type = self.context.f64_type();
        let intrinsic_func = intrinsic
            .get_declaration(&self.module, &[f64_type.into()])
            .ok_or_else(|| format!("Failed to get {} declaration", intrinsic_name))?;

        // Create a wrapper function with internal linkage (avoids duplicate symbol with runtime)
        let wrapper = self.module.add_function(
            func_name,
            *fn_type,
            Some(inkwell::module::Linkage::Internal),
        );
        wrapper.add_attribute(
            inkwell::attributes::AttributeLoc::Function,
            self.context.create_enum_attribute(
                inkwell::attributes::Attribute::get_named_enum_kind_id("alwaysinline"),
                0,
            ),
        );

        let bb = self.context.append_basic_block(wrapper, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(bb);

        // Collect non-void args (the intrinsic takes only the f64 args, not env)
        let params: Vec<inkwell::values::BasicMetadataValueEnum> = wrapper
            .get_params()
            .into_iter()
            .filter(|p| p.is_float_value())
            .map(|p| p.into())
            .collect();

        let result = builder
            .build_call(intrinsic_func, &params, "result")
            .map_err(|e| format!("Failed to build intrinsic call: {}", e))?
            .try_as_basic_value()
            .left()
            .ok_or("Intrinsic returned void unexpectedly")?;

        builder
            .build_return(Some(&result))
            .map_err(|e| format!("Failed to build return: {}", e))?;

        Ok(Some(wrapper))
    }

    /// Create inline Std intrinsics (e.g., Std.int for float-to-int conversion).
    /// Returns Some(func) if replaced, None if not a known Std function.
    fn try_create_std_intrinsic(
        &self,
        func_name: &str,
        fn_type: &inkwell::types::FunctionType<'ctx>,
    ) -> Result<Option<FunctionValue<'ctx>>, String> {
        if func_name != "haxe_std_int" {
            return Ok(None);
        }

        // Create a wrapper function with internal linkage
        let wrapper = self.module.add_function(
            func_name,
            *fn_type,
            Some(inkwell::module::Linkage::Internal),
        );
        wrapper.add_attribute(
            inkwell::attributes::AttributeLoc::Function,
            self.context.create_enum_attribute(
                inkwell::attributes::Attribute::get_named_enum_kind_id("alwaysinline"),
                0,
            ),
        );

        let bb = self.context.append_basic_block(wrapper, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(bb);

        let i32_type = self.context.i32_type();

        // Get the first non-pointer parameter (could be float or int)
        let param = wrapper
            .get_params()
            .into_iter()
            .find(|p| p.is_float_value() || p.is_int_value())
            .ok_or("haxe_std_int: expected numeric parameter")?;

        // Determine the actual return type from the function signature
        let ret_type = fn_type
            .get_return_type()
            .map(|t| t.into_int_type())
            .unwrap_or(i32_type);

        let i32_result: inkwell::values::IntValue = if param.is_float_value() {
            // Convert f64 to i32 using fptosi (truncation toward zero)
            builder
                .build_float_to_signed_int(param.into_float_value(), i32_type, "int_result")
                .map_err(|e| format!("Failed to build fptosi: {}", e))?
        } else {
            // Already an int, truncate/extend to i32 if needed
            let int_val = param.into_int_value();
            let int_bits = int_val.get_type().get_bit_width();
            if int_bits > 32 {
                builder
                    .build_int_truncate(int_val, i32_type, "trunc")
                    .map_err(|e| format!("Failed to truncate: {}", e))?
            } else if int_bits < 32 {
                builder
                    .build_int_s_extend(int_val, i32_type, "sext")
                    .map_err(|e| format!("Failed to sign-extend: {}", e))?
            } else {
                int_val
            }
        };

        // Sign-extend result to match the function's return type if needed
        let result = if ret_type.get_bit_width() > 32 {
            builder
                .build_int_s_extend(i32_result, ret_type, "sext_ret")
                .map_err(|e| format!("Failed to sign-extend return: {}", e))?
        } else {
            i32_result
        };

        builder
            .build_return(Some(&result))
            .map_err(|e| format!("Failed to build return: {}", e))?;

        Ok(Some(wrapper))
    }

    /// Create inline array intrinsics for common array operations.
    /// HaxeArray layout: { ptr: *mut u8 (0), len: usize (8), cap: usize (16), elem_size: usize (24) }
    fn try_create_array_intrinsic(
        &self,
        func_name: &str,
        fn_type: &inkwell::types::FunctionType<'ctx>,
    ) -> Result<Option<FunctionValue<'ctx>>, String> {
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        match func_name {
            // haxe_array_length(arr_ptr) -> i64
            // Returns the length field at offset 8
            "haxe_array_length" => {
                let wrapper = self.module.add_function(
                    func_name,
                    *fn_type,
                    Some(inkwell::module::Linkage::Internal),
                );
                wrapper.add_attribute(
                    inkwell::attributes::AttributeLoc::Function,
                    self.context.create_enum_attribute(
                        inkwell::attributes::Attribute::get_named_enum_kind_id("alwaysinline"),
                        0,
                    ),
                );

                let bb = self.context.append_basic_block(wrapper, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);

                let arr_ptr = wrapper.get_first_param().ok_or("Missing arr_ptr param")?;
                let arr_ptr = arr_ptr.into_pointer_value();

                // Load length from offset 8
                let len_ptr = unsafe {
                    builder
                        .build_gep(
                            i64_type,
                            arr_ptr,
                            &[i64_type.const_int(1, false)],
                            "len_ptr",
                        )
                        .map_err(|e| format!("GEP failed: {}", e))?
                };
                let len = builder
                    .build_load(i64_type, len_ptr, "len")
                    .map_err(|e| format!("Load failed: {}", e))?;

                builder
                    .build_return(Some(&len))
                    .map_err(|e| format!("Return failed: {}", e))?;

                Ok(Some(wrapper))
            }

            // haxe_array_get_ptr(arr_ptr, index) -> ptr
            // Returns pointer to element: data_ptr + index * elem_size
            "haxe_array_get_ptr" => {
                let wrapper = self.module.add_function(
                    func_name,
                    *fn_type,
                    Some(inkwell::module::Linkage::Internal),
                );
                wrapper.add_attribute(
                    inkwell::attributes::AttributeLoc::Function,
                    self.context.create_enum_attribute(
                        inkwell::attributes::Attribute::get_named_enum_kind_id("alwaysinline"),
                        0,
                    ),
                );

                let bb = self.context.append_basic_block(wrapper, "entry");
                let builder = self.context.create_builder();
                builder.position_at_end(bb);

                let params: Vec<_> = wrapper.get_params();
                if params.len() < 2 {
                    return Err("haxe_array_get_ptr requires 2 params".to_string());
                }
                let arr_ptr = params[0].into_pointer_value();
                let index = params[1].into_int_value();

                // Load data_ptr from offset 0
                let data_ptr = builder
                    .build_load(ptr_type, arr_ptr, "data_ptr")
                    .map_err(|e| format!("Load data_ptr failed: {}", e))?
                    .into_pointer_value();

                // Load elem_size from offset 24 (3 * 8 bytes)
                let elem_size_ptr = unsafe {
                    builder
                        .build_gep(
                            i64_type,
                            arr_ptr,
                            &[i64_type.const_int(3, false)],
                            "elem_size_ptr",
                        )
                        .map_err(|e| format!("GEP elem_size failed: {}", e))?
                };
                let elem_size = builder
                    .build_load(i64_type, elem_size_ptr, "elem_size")
                    .map_err(|e| format!("Load elem_size failed: {}", e))?
                    .into_int_value();

                // Compute byte_offset = index * elem_size
                let byte_offset = builder
                    .build_int_mul(index, elem_size, "byte_offset")
                    .map_err(|e| format!("Mul failed: {}", e))?;

                // Compute element pointer = data_ptr + byte_offset
                let i8_type = self.context.i8_type();
                let elem_ptr = unsafe {
                    builder
                        .build_gep(i8_type, data_ptr, &[byte_offset], "elem_ptr")
                        .map_err(|e| format!("GEP elem_ptr failed: {}", e))?
                };

                builder
                    .build_return(Some(&elem_ptr))
                    .map_err(|e| format!("Return failed: {}", e))?;

                Ok(Some(wrapper))
            }

            _ => Ok(None),
        }
    }

    /// Get or create an LLVM global variable for inline global access.
    /// This eliminates FFI calls to rayzor_global_load/store.
    fn get_or_create_global(&mut self, global_id: IrGlobalId) -> GlobalValue<'ctx> {
        let idx = global_id.0 as usize;

        // Ensure Vec is large enough
        if idx >= self.global_vars.len() {
            self.global_vars.resize(idx + 1, None);
        }

        // Return existing global if present
        if let Some(global) = self.global_vars[idx] {
            return global;
        }

        // Create a new LLVM global variable (i64 to hold any value type)
        let global_name = format!("__rayzor_global_{}", global_id.0);
        let i64_type = self.context.i64_type();

        let global = self.module.add_global(i64_type, None, &global_name);
        global.set_initializer(&i64_type.const_zero());
        global.set_linkage(inkwell::module::Linkage::Internal);

        self.global_vars[idx] = Some(global);
        global
    }

    /// Compile function bodies for a module (call declare_module for ALL modules first)
    pub fn compile_module_bodies(&mut self, module: &IrModule) -> Result<(), String> {
        for (func_id, function) in &module.functions {
            let llvm_func = *self
                .function_map
                .get(func_id)
                .ok_or_else(|| format!("Function {:?} not declared", func_id))?;

            // Skip if already compiled
            if llvm_func.count_basic_blocks() > 0 {
                continue;
            }

            // Skip extern functions (no body)
            if function.cfg.blocks.is_empty() {
                continue;
            }

            self.compile_function_body(*func_id, function, llvm_func)
                .map_err(|e| {
                    format!(
                        "Error in function '{}' ({:?}): {}",
                        function.name, func_id, e
                    )
                })?;

            // Per-function verification to catch return type mismatches early
            if !llvm_func.verify(true) {
                // verify(true) prints the error to stderr
                return Err(format!(
                    "LLVM verification failed for function '{}' ({:?}). Check stderr for details.",
                    function.name, func_id,
                ));
            }
        }
        Ok(())
    }

    /// Compile an entire MIR module
    ///
    /// This is the main entry point for whole-program compilation.
    /// Compiles all functions and registers runtime symbols.
    ///
    /// Note: If compiling multiple modules with cross-module references,
    /// use declare_module() for ALL modules first, then compile_module_bodies().
    pub fn compile_module(&mut self, module: &IrModule) -> Result<(), String> {
        // Note: We don't pre-declare runtime symbols as functions here.
        // They will be declared when encountered in MIR code with proper signatures,
        // and resolved at link time via add_global_mapping in finalize().

        // Phase 1: Declare all functions first (forward declarations)
        self.declare_module(module)?;

        // Phase 2: Compile all function bodies
        self.compile_module_bodies(module)?;

        // Note: Execution engine is created lazily when first needed (call_main, get_function_ptr, etc.)
        // This allows compiling multiple modules before JIT-compiling everything
        Ok(())
    }

    /// Finalize compilation and create execution engine
    /// Call this after all modules have been compiled
    pub fn finalize(&mut self) -> Result<(), String> {
        if self.execution_engine.is_some() {
            return Ok(()); // Already finalized
        }

        // Dump IR before optimization if requested
        if std::env::var("RAYZOR_DUMP_LLVM_IR").is_ok() {
            let ir_str = self.module.print_to_string().to_string();
            // Save to file for easier inspection
            if let Ok(_) = std::fs::write("/tmp/rayzor_llvm_ir.ll", &ir_str) {
                eprintln!(
                    "=== LLVM IR saved to /tmp/rayzor_llvm_ir.ll ({} bytes) ===",
                    ir_str.len()
                );
            } else {
                eprintln!(
                    "=== LLVM IR (before JIT, opt_level={:?}) ===",
                    self.opt_level
                );
                // Fallback to truncated output
                if ir_str.len() > 5000 {
                    eprintln!("{}...(truncated)", &ir_str[..5000]);
                } else {
                    eprintln!("{}", ir_str);
                }
                eprintln!("=== End LLVM IR ===");
            }
        }

        // Verify the module before optimization
        if let Err(msg) = self.module.verify() {
            // Print the module IR for debugging
            if std::env::var("RAYZOR_DUMP_LLVM_IR").is_ok() {
                eprintln!("=== LLVM IR (verification failed) ===");
                eprintln!("{}", self.module.print_to_string().to_string());
                eprintln!("=== End LLVM IR ===");
            }
            return Err(format!(
                "LLVM module verification failed: {}",
                msg.to_string()
            ));
        }

        // Run LLVM optimization passes before JIT compilation
        // This is critical for performance - without this, we're running unoptimized IR
        if self.opt_level != OptimizationLevel::None {
            let passes = match self.opt_level {
                OptimizationLevel::None => "default<O0>",
                OptimizationLevel::Less => "default<O1>",
                OptimizationLevel::Default => "default<O2>",
                OptimizationLevel::Aggressive => "default<O3>",
            };

            // Get target machine for optimization
            let target_triple = TargetMachine::get_default_triple();
            let target = Target::from_triple(&target_triple)
                .map_err(|e| format!("Failed to get target: {}", e))?;
            let target_machine = target
                .create_target_machine(
                    &target_triple,
                    TargetMachine::get_host_cpu_name()
                        .to_str()
                        .unwrap_or("generic"),
                    TargetMachine::get_host_cpu_features()
                        .to_str()
                        .unwrap_or(""),
                    self.opt_level,
                    RelocMode::Default,
                    CodeModel::Default,
                )
                .ok_or("Failed to create target machine for optimization")?;

            // Run optimization passes with tuned options for x86_64 performance
            let pass_options = Self::create_pass_options();
            self.module
                .run_passes(passes, &target_machine, pass_options)
                .map_err(|e| format!("Failed to run optimization passes: {}", e))?;
        }

        // Dump optimized LLVM IR for debugging if requested
        if std::env::var("RAYZOR_DUMP_LLVM_IR").is_ok() {
            let ir_str = self.module.print_to_string().to_string();
            if std::fs::write("/tmp/rayzor_llvm_ir_opt.ll", &ir_str).is_ok() {
                eprintln!(
                    "=== Optimized LLVM IR saved to /tmp/rayzor_llvm_ir_opt.ll ({} bytes) ===",
                    ir_str.len()
                );
            }
        }

        // Verify module before JIT compilation
        if let Err(msg) = self.module.verify() {
            return Err(format!(
                "LLVM module verification failed: {}",
                msg.to_string()
            ));
        }

        // Register runtime symbols via LLVM's DynamicLibrary (process-level symbol table)
        // BEFORE creating the execution engine. MCJIT resolves symbols via RuntimeDyld
        // during module loading, so all symbols must be available before engine creation
        // to prevent RuntimeDyld from hitting unresolved symbols and crashing in
        // resolveRelocations with a null mutex (intermittent ~33% failure rate on Linux).
        for (name, addr) in &self.runtime_symbols {
            let c_name = std::ffi::CString::new(name.as_str()).unwrap();
            unsafe {
                llvm_sys::support::LLVMAddSymbol(c_name.as_ptr(), *addr as *mut std::ffi::c_void);
            }
        }

        // Create execution engine with full optimization
        // The execution engine needs the opt_level for machine code generation (instruction
        // selection, register allocation, etc.), separate from the IR-level optimizations
        // that run_passes() already performed.
        let engine = self
            .module
            .create_jit_execution_engine(self.opt_level)
            .map_err(|e| format!("Failed to create JIT execution engine: {}", e))?;

        // Also add global mappings as a fallback for symbols that MCJIT
        // might resolve through the execution engine rather than RuntimeDyld
        let mut mapped_count = 0;
        let mut unmapped: Vec<String> = Vec::new();
        for (name, addr) in &self.runtime_symbols {
            if let Some(func) = self.module.get_function(name) {
                engine.add_global_mapping(&func, *addr);
                mapped_count += 1;
            } else {
                unmapped.push(name.clone());
            }
        }
        if !unmapped.is_empty() {
            tracing::debug!(
                "[LLVM] Mapped {}/{} runtime symbols. Unmapped: {:?}",
                mapped_count,
                self.runtime_symbols.len(),
                unmapped
            );
        }

        self.execution_engine = Some(engine);

        // NOTE: We skip pre-compilation of functions here.
        // LLVM's MCJIT does lazy compilation when get_function_address is called.
        // Pre-compilation was causing intermittent segfaults in LLVM's MCJIT (~40% failure rate).
        // Instead, we get function pointers lazily in get_function_pointer() and get_all_function_pointers().
        //
        // The tradeoff is that first execution of each function has JIT overhead,
        // but this is much safer than triggering LLVM bugs during bulk pre-compilation.

        Ok(())
    }

    /// Compile to object file for AOT/dylib compilation
    ///
    /// This avoids all MCJIT memory protection issues on Apple Silicon by:
    /// 1. Writing an object file to disk
    /// 2. Linking with the system linker (which handles MAP_JIT correctly)
    /// 3. Loading the resulting dylib with dlopen (no JIT needed)
    ///
    /// Returns the path to the generated object file.
    pub fn compile_to_object_file(&mut self, output_path: &std::path::Path) -> Result<(), String> {
        // Dump IR for debugging if requested
        if std::env::var("RAYZOR_DUMP_LLVM_IR").is_ok() {
            let ir_str = self.module.print_to_string().to_string();
            if std::fs::write("/tmp/rayzor_llvm_ir.ll", &ir_str).is_ok() {
                eprintln!(
                    "=== LLVM IR saved to /tmp/rayzor_llvm_ir.ll ({} bytes) ===",
                    ir_str.len()
                );
            }
        }

        // Verify module before compilation
        if let Err(msg) = self.module.verify() {
            return Err(format!(
                "LLVM module verification failed: {}",
                msg.to_string()
            ));
        }

        // Get target machine with PIC mode for shared library
        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple)
            .map_err(|e| format!("Failed to get target: {}", e))?;

        let target_machine = target
            .create_target_machine(
                &target_triple,
                TargetMachine::get_host_cpu_name()
                    .to_str()
                    .unwrap_or("generic"),
                TargetMachine::get_host_cpu_features()
                    .to_str()
                    .unwrap_or(""),
                self.opt_level,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or("Failed to create target machine for AOT")?;

        // Run optimization passes
        if self.opt_level != OptimizationLevel::None {
            let passes = match self.opt_level {
                OptimizationLevel::None => "default<O0>",
                OptimizationLevel::Less => "default<O1>",
                OptimizationLevel::Default => "default<O2>",
                OptimizationLevel::Aggressive => "default<O3>",
            };
            let pass_options = Self::create_pass_options();
            self.module
                .run_passes(passes, &target_machine, pass_options)
                .map_err(|e| format!("Failed to run optimization passes: {}", e))?;
        }

        // Write object file
        target_machine
            .write_to_file(&self.module, FileType::Object, output_path)
            .map_err(|e| format!("Failed to write object file: {}", e))?;

        Ok(())
    }

    /// Get a reference to the underlying LLVM module (for AOT operations).
    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Get all function symbol names that will be exported in the dylib
    ///
    /// Returns a map of IrFunctionId -> symbol name for loading from the dylib
    pub fn get_function_symbols(&self) -> HashMap<IrFunctionId, String> {
        self.function_map
            .iter()
            .filter(|(id, _)| !self.extern_function_ids.contains(id))
            .map(|(id, func)| (*id, func.get_name().to_string_lossy().to_string()))
            .collect()
    }

    /// Declare an external function for FFI
    fn declare_external_function(&self, name: &str) -> Result<FunctionValue<'ctx>, String> {
        // Check if already declared
        if let Some(func) = self.module.get_function(name) {
            return Ok(func);
        }

        // Declare as variadic function returning void for flexibility
        // Runtime functions have varying signatures; LLVM will handle the ABI
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[], true); // true = variadic

        Ok(self.module.add_function(name, fn_type, None))
    }

    /// Call main function in the module
    pub fn call_main(&mut self, module: &IrModule) -> Result<(), String> {
        // Finalize if needed
        self.finalize()?;

        // Find main function by name since IDs may not match between modules
        let main_func = module
            .functions
            .iter()
            .find(|(_, f)| f.name.ends_with("_main") || f.name == "main")
            .map(|(_, f)| f)
            .ok_or("No main function found")?;

        // Get function pointer by name (MCJIT compilation already happened in finalize)
        let func_name = Self::mangle_function_name(&main_func.name);
        let engine = self
            .execution_engine
            .as_ref()
            .ok_or("Execution engine not initialized")?;

        let fn_ptr = engine
            .get_function_address(&func_name)
            .map_err(|e| format!("Failed to get main function '{}': {}", func_name, e))?;

        unsafe {
            // Haxe functions have a hidden environment parameter (i64) that must be passed
            // even if not used. Pass null (0) for the environment pointer.
            // This matches the Cranelift backend's calling convention.
            let main_fn: extern "C" fn(i64) = std::mem::transmute(fn_ptr);
            main_fn(0); // null environment pointer
        }

        Ok(())
    }

    /// Translate function type signature to LLVM function type
    fn translate_function_type(
        &self,
        ty: &IrType,
    ) -> Result<inkwell::types::FunctionType<'ctx>, String> {
        match ty {
            IrType::Function {
                params,
                return_type,
                ..
            } => {
                // Translate parameter types
                let param_types: Result<Vec<BasicMetadataTypeEnum>, _> = params
                    .iter()
                    .map(|param_ty| self.translate_type(param_ty).map(|t| t.into()))
                    .collect();
                let param_types = param_types?;

                // Translate return type
                if **return_type == IrType::Void {
                    Ok(self.context.void_type().fn_type(&param_types, false))
                } else {
                    let ret_ty = self.translate_type(return_type)?;
                    Ok(ret_ty.fn_type(&param_types, false))
                }
            }
            _ => Err(format!("Expected function type, got {:?}", ty)),
        }
    }

    /// Translate MIR type to LLVM type
    fn translate_type(&self, ty: &IrType) -> Result<BasicTypeEnum<'ctx>, String> {
        match ty {
            IrType::Void => Err(format!(
                "Void type cannot be used as BasicType (in {:?})",
                ty
            )),
            IrType::Bool | IrType::I8 | IrType::U8 => Ok(self.context.i8_type().into()),
            IrType::I16 | IrType::U16 => Ok(self.context.i16_type().into()),
            IrType::I32 | IrType::U32 => Ok(self.context.i32_type().into()),
            IrType::I64 | IrType::U64 => Ok(self.context.i64_type().into()),
            IrType::F32 => Ok(self.context.f32_type().into()),
            IrType::F64 => Ok(self.context.f64_type().into()),

            // Pointers become opaque pointers in LLVM 15+
            IrType::Ptr(_) | IrType::Ref(_) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }

            // Arrays
            IrType::Array(elem_ty, count) => {
                let elem_llvm_ty = self.translate_type(elem_ty)?;
                Ok(elem_llvm_ty.array_type(*count as u32).into())
            }

            // Slices are represented as {ptr, len}
            IrType::Slice(_) => {
                let ptr_ty = self.context.ptr_type(AddressSpace::default());
                let len_ty = self.context.i64_type();
                Ok(self
                    .context
                    .struct_type(&[ptr_ty.into(), len_ty.into()], false)
                    .into())
            }

            // Strings are heap-allocated HaxeString* pointers
            IrType::String => Ok(self.context.ptr_type(AddressSpace::default()).into()),

            // Functions become function pointers
            IrType::Function { .. } => Ok(self.context.ptr_type(AddressSpace::default()).into()),

            // Structs
            IrType::Struct { fields, .. } => {
                let mut field_types = Vec::new();
                for f in fields {
                    // Skip void fields (they have no size)
                    if f.ty == IrType::Void {
                        continue;
                    }
                    field_types.push(self.translate_type(&f.ty)?);
                }
                Ok(self.context.struct_type(&field_types, false).into())
            }

            // Unions are represented as a struct with a tag + largest variant
            IrType::Union { variants, .. } => {
                let tag_ty = self.context.i32_type();

                // Find largest variant size
                let mut max_size = 0usize;
                for variant in variants {
                    let size: usize = variant.fields.iter().map(|f| f.size()).sum();
                    max_size = max_size.max(size);
                }

                // Create union as {i32 tag, [i8 x max_size]}
                let data_ty = self.context.i8_type().array_type(max_size as u32);
                Ok(self
                    .context
                    .struct_type(&[tag_ty.into(), data_ty.into()], false)
                    .into())
            }

            IrType::Opaque { size, .. } => {
                // Opaque types become byte arrays
                Ok(self.context.i8_type().array_type(*size as u32).into())
            }

            IrType::Any => {
                // Any type is an opaque pointer in LLVM
                // This avoids costly ptrtoint/inttoptr conversions that break LLVM optimizations.
                // Pointers pass through directly; integers use inttoptr when needed.
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }

            IrType::TypeVar(_) => Ok(self.context.i64_type().into()), // Safety net: pointer-sized

            IrType::Generic { .. } => {
                Err("Generic types should be monomorphized before codegen".to_string())
            }

            // SIMD Vector types - translate to LLVM vector types
            IrType::Vector { element, count } => {
                match (element.as_ref(), *count) {
                    // 128-bit float vectors
                    (IrType::F32, 4) => Ok(self.context.f32_type().vec_type(4).into()),
                    (IrType::F64, 2) => Ok(self.context.f64_type().vec_type(2).into()),

                    // 128-bit integer vectors
                    (IrType::I8 | IrType::U8, 16) => Ok(self.context.i8_type().vec_type(16).into()),
                    (IrType::I16 | IrType::U16, 8) => {
                        Ok(self.context.i16_type().vec_type(8).into())
                    }
                    (IrType::I32 | IrType::U32, 4) => {
                        Ok(self.context.i32_type().vec_type(4).into())
                    }
                    (IrType::I64 | IrType::U64, 2) => {
                        Ok(self.context.i64_type().vec_type(2).into())
                    }

                    // Generic vector type support
                    _ => {
                        let elem_ty = self.translate_type(element)?;
                        match elem_ty {
                            BasicTypeEnum::IntType(int_ty) => {
                                Ok(int_ty.vec_type(*count as u32).into())
                            }
                            BasicTypeEnum::FloatType(float_ty) => {
                                Ok(float_ty.vec_type(*count as u32).into())
                            }
                            _ => Err(format!("Unsupported vector element type: {:?}", element)),
                        }
                    }
                }
            }
        }
    }

    /// Declare a function signature
    ///
    /// Uses the function's unique name (not just ID) to avoid collisions when
    /// compiling multiple modules with overlapping IrFunctionIds.
    fn declare_function(
        &mut self,
        func_id: IrFunctionId,
        function: &IrFunction,
    ) -> Result<FunctionValue<'ctx>, String> {
        // Use function's actual name for LLVM (unique across modules)
        // Mangle the name to be LLVM-safe (replace :: with _)
        let func_name = Self::mangle_function_name(&function.name);

        // If this function is ExternC or uses C calling convention (no env param), treat it as extern
        let is_c_abi = function.kind == crate::ir::functions::FunctionKind::ExternC
            || function.signature.calling_convention == crate::ir::CallingConvention::C;
        if is_c_abi {
            self.extern_function_ids.insert(func_id);

            // Translate parameter types (NO env param for extern C functions)
            let param_types: Result<Vec<BasicMetadataTypeEnum>, _> = function
                .signature
                .parameters
                .iter()
                .filter(|param| param.ty != IrType::Void)
                .map(|param| self.translate_type(&param.ty).map(|t| t.into()))
                .collect();
            let param_types = param_types?;

            let fn_type = if function.signature.return_type == IrType::Void {
                self.context.void_type().fn_type(&param_types, false)
            } else {
                let return_type = self.translate_type(&function.signature.return_type)?;
                return_type.fn_type(&param_types, false)
            };

            // Replace known math runtime functions with LLVM intrinsic wrappers
            // (e.g. haxe_math_sqrt → @llvm.sqrt.f64 → single fsqrt instruction)
            if let Some(llvm_func) = self.try_create_math_intrinsic(&func_name, &fn_type)? {
                self.function_map.insert(func_id, llvm_func);
                return Ok(llvm_func);
            }

            // Replace Std functions with inline implementations (e.g., Std.int → fptosi)
            if let Some(llvm_func) = self.try_create_std_intrinsic(&func_name, &fn_type)? {
                self.function_map.insert(func_id, llvm_func);
                return Ok(llvm_func);
            }

            // Replace array operations with inline implementations
            // (e.g. haxe_array_length → inline load from offset 8)
            if let Some(llvm_func) = self.try_create_array_intrinsic(&func_name, &fn_type)? {
                self.function_map.insert(func_id, llvm_func);
                return Ok(llvm_func);
            }

            // Check if already declared
            if let Some(existing_func) = self.module.get_function(&func_name) {
                let existing_params = existing_func.get_type().get_param_types();
                if existing_params.len() == param_types.len() {
                    self.function_map.insert(func_id, existing_func);
                    return Ok(existing_func);
                }
                // Signature mismatch - create with unique name
                let unique_name = format!("{}__extern_{}", func_name, func_id.0);
                let llvm_func = self.module.add_function(
                    &unique_name,
                    fn_type,
                    Some(inkwell::module::Linkage::External),
                );
                self.function_map.insert(func_id, llvm_func);
                return Ok(llvm_func);
            }

            let llvm_func = self.module.add_function(
                &func_name,
                fn_type,
                Some(inkwell::module::Linkage::External),
            );
            self.function_map.insert(func_id, llvm_func);
            return Ok(llvm_func);
        }

        // Check if this function was already declared (from a previous module)
        // Reuse if it has basic blocks (was already compiled) AND signatures match
        if let Some(existing_func) = self.module.get_function(&func_name) {
            // Build expected signature to compare
            let expected_param_types: Result<Vec<BasicMetadataTypeEnum>, _> = function
                .signature
                .parameters
                .iter()
                .filter(|param| param.ty != IrType::Void)
                .map(|param| self.translate_type(&param.ty).map(|t| t.into()))
                .collect();
            let expected_params = expected_param_types?;

            let existing_type = existing_func.get_type();
            let existing_params: Vec<BasicMetadataTypeEnum> = existing_type
                .get_param_types()
                .iter()
                .map(|&t| t.into())
                .collect();

            // Check if signatures match (parameter count and types)
            let signatures_match = expected_params.len() == existing_params.len()
                && expected_params
                    .iter()
                    .zip(existing_params.iter())
                    .all(|(e, a)| {
                        // Compare type kinds - a simple check
                        format!("{:?}", e) == format!("{:?}", a)
                    });

            if signatures_match {
                // Signatures match, safe to reuse
                self.function_map.insert(func_id, existing_func);
                // Still need to track sret for this func_id even when reusing
                if function.signature.uses_sret {
                    self.sret_function_ids.insert(func_id);
                }
                return Ok(existing_func);
            } else {
                // Signature mismatch - this is a different function with same name
                // Generate a unique name using the func_id to disambiguate
                let unique_name = format!("{}_{}", func_name, func_id.0);
                if let Some(unique_func) = self.module.get_function(&unique_name) {
                    self.function_map.insert(func_id, unique_func);
                    // Track sret for this func_id
                    if function.signature.uses_sret {
                        self.sret_function_ids.insert(func_id);
                    }
                    return Ok(unique_func);
                }
                // Fall through to create new function with unique name
                return self.declare_function_with_name(func_id, function, &unique_name);
            }
        }

        // Translate parameter types
        // IMPORTANT: Match Cranelift's calling convention for Haxe functions
        // Order of hidden parameters:
        // 1. sret pointer (if uses_sret) - struct return via hidden pointer
        // 2. env parameter (i64) - environment/closure pointer
        // 3. actual user parameters
        let mut param_types: Vec<BasicMetadataTypeEnum> = Vec::new();

        // Check if this function uses sret (struct return by hidden pointer)
        let uses_sret = function.signature.uses_sret;
        if uses_sret {
            // Track this function as using sret
            self.sret_function_ids.insert(func_id);
            // Add sret pointer parameter (ptr type) as first hidden param
            param_types.push(
                self.context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
            );
        }

        // Add hidden env parameter (i64)
        param_types.push(self.context.i64_type().into());

        // Then add actual IR parameters
        for param in &function.signature.parameters {
            if param.ty != IrType::Void {
                let ty = self.translate_type(&param.ty)?;
                param_types.push(ty.into());
            }
        }

        // Translate return type
        // If using sret, the function returns void (value written through sret pointer)
        let fn_type = if uses_sret || function.signature.return_type == IrType::Void {
            // Void function (sret functions also return void)
            self.context.void_type().fn_type(&param_types, false)
        } else {
            // Function with return value
            let return_type = self.translate_type(&function.signature.return_type)?;
            return_type.fn_type(&param_types, false)
        };

        let llvm_func = self.module.add_function(&func_name, fn_type, None);

        self.function_map.insert(func_id, llvm_func);
        Ok(llvm_func)
    }

    /// Mangle function name to be LLVM-safe
    pub fn mangle_function_name(name: &str) -> String {
        // Replace characters that might cause issues in LLVM
        name.replace("::", "_")
            .replace('<', "_L_")
            .replace('>', "_R_")
            .replace(',', "_C_")
            .replace(' ', "_S_")
    }

    /// Declare a function with a specific name (for handling signature conflicts)
    fn declare_function_with_name(
        &mut self,
        func_id: IrFunctionId,
        function: &IrFunction,
        func_name: &str,
    ) -> Result<FunctionValue<'ctx>, String> {
        // Translate parameter types with hidden params first
        // Order: sret (if needed), env, user params
        let mut param_types: Vec<BasicMetadataTypeEnum> = Vec::new();

        // Check if this function uses sret (struct return by hidden pointer)
        let uses_sret = function.signature.uses_sret;
        if uses_sret {
            // Track this function as using sret
            self.sret_function_ids.insert(func_id);
            // Add sret pointer parameter (ptr type) as first hidden param
            param_types.push(
                self.context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
            );
        }

        // Add hidden env parameter (i64)
        param_types.push(self.context.i64_type().into());

        // Then add actual IR parameters
        for param in &function.signature.parameters {
            if param.ty != IrType::Void {
                let ty = self.translate_type(&param.ty)?;
                param_types.push(ty.into());
            }
        }

        // Translate return type (void if uses_sret)
        let fn_type = if uses_sret || function.signature.return_type == IrType::Void {
            self.context.void_type().fn_type(&param_types, false)
        } else {
            let return_type = self.translate_type(&function.signature.return_type)?;
            return_type.fn_type(&param_types, false)
        };

        let llvm_func = self.module.add_function(func_name, fn_type, None);
        self.function_map.insert(func_id, llvm_func);
        Ok(llvm_func)
    }

    /// Compile function body
    fn compile_function_body(
        &mut self,
        func_id: IrFunctionId,
        function: &IrFunction,
        llvm_func: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        // Clear previous compilation state
        self.value_map.clear();
        self.block_map.clear();
        self.phi_map.clear();
        self.alloca_ids.clear();
        self.current_sret_ptr = None;

        // Check if this function uses sret (struct return)
        let uses_sret = self.sret_function_ids.contains(&func_id);
        // Check if this function uses C calling convention (no hidden env param)
        let is_c_abi = self.extern_function_ids.contains(&func_id);

        // Map function parameters to LLVM values using their actual IrIds
        // Note: we filter out void parameters but need to handle IrIds correctly
        let non_void_params: Vec<_> = function
            .signature
            .parameters
            .iter()
            .filter(|p| p.ty != IrType::Void)
            .collect();

        // Debug: Log parameter mapping for troubleshooting
        if cfg!(debug_assertions) {
            let param_ids: Vec<_> = function
                .signature
                .parameters
                .iter()
                .map(|p| (p.reg.as_u32(), &p.ty))
                .collect();
            tracing::debug!(
                "Function '{}': parameters {:?}, uses_sret: {}, is_c_abi: {}",
                function.name,
                param_ids,
                uses_sret,
                is_c_abi
            );
        }

        // First, map void parameters to a placeholder value (they shouldn't be used)
        for param in &function.signature.parameters {
            if param.ty == IrType::Void {
                // Insert a null pointer as placeholder - it shouldn't be used
                let placeholder = self.context.i8_type().const_int(0, false).into();
                self.value_map.insert(param.reg, placeholder);
            }
        }

        // Calculate the offset for IR parameters based on hidden params:
        // - C ABI (is_c_abi): no hidden params, param_offset = 0
        // - Haxe with sret: param 0 = sret ptr, param 1 = env, params 2+ = IR params
        // - Haxe no sret: param 0 = env, params 1+ = IR params
        let param_offset = if is_c_abi {
            0 // C ABI: no hidden parameters
        } else if uses_sret {
            2 // Haxe with sret: sret + env
        } else {
            1 // Haxe: just env
        };

        // Then map non-void parameters to their LLVM values
        for (i, llvm_param) in llvm_func.get_param_iter().enumerate() {
            if !is_c_abi && uses_sret && i == 0 {
                // Capture the sret pointer for use in Return terminator (Haxe ABI only)
                self.current_sret_ptr = Some(llvm_param.into_pointer_value());
                continue;
            }
            if !is_c_abi && i == (if uses_sret { 1 } else { 0 }) {
                // Skip the hidden env parameter (Haxe ABI only)
                continue;
            }
            let ir_param_idx = i - param_offset;
            if ir_param_idx < non_void_params.len() {
                let param_id = non_void_params[ir_param_idx].reg;
                self.value_map.insert(param_id, llvm_param);
            }
        }

        // Also, map any local variables that are used before being defined
        // This is a workaround for MIR patterns where locals are referenced early
        for (local_id, local) in &function.locals {
            if !self.value_map.contains_key(local_id) {
                // Insert a default value based on type
                let default = match &local.ty {
                    IrType::Void => continue,
                    IrType::Bool => self.context.i8_type().const_int(0, false).into(), // Bool is i8
                    IrType::I8 | IrType::U8 => self.context.i8_type().const_int(0, false).into(),
                    IrType::I16 | IrType::U16 => self.context.i16_type().const_int(0, false).into(),
                    IrType::I32 | IrType::U32 => self.context.i32_type().const_int(0, false).into(),
                    IrType::I64 | IrType::U64 => self.context.i64_type().const_int(0, false).into(),
                    IrType::F32 => self.context.f32_type().const_float(0.0).into(),
                    IrType::F64 => self.context.f64_type().const_float(0.0).into(),
                    _ => self.context.i64_type().const_int(0, false).into(),
                };
                self.value_map.insert(*local_id, default);
            }
        }

        // Create LLVM basic blocks for all MIR blocks
        // IMPORTANT: In LLVM, the entry block cannot have predecessors (no branches to it).
        // We create a "true entry" block that just branches to the first MIR block.
        // This ensures loops back to block 0 don't violate LLVM's entry block rule.

        // Get blocks in reverse post-order (RPO) to ensure definitions are
        // visited before uses. This is critical after inlining creates blocks
        // with higher IDs in the middle of control flow.
        let sorted_blocks = {
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

        // Create the LLVM entry block (will branch to first MIR block)
        let entry_block = self.context.append_basic_block(llvm_func, "entry");

        // Create LLVM blocks for all MIR blocks
        for (block_id, _) in &function.cfg.blocks {
            let block_name = format!("bb{}", block_id.as_u32());
            let llvm_block = self.context.append_basic_block(llvm_func, &block_name);
            self.block_map.insert(*block_id, llvm_block);
        }

        // Connect entry block to first MIR block (bb0)
        // Get the first MIR block ID (should be block 0 in sorted order)
        if let Some((first_block_id, _)) = sorted_blocks.first() {
            let first_mir_block = self.block_map[first_block_id];
            self.builder.position_at_end(entry_block);
            self.builder
                .build_unconditional_branch(first_mir_block)
                .map_err(|e| format!("Failed to build entry branch: {}", e))?;
        }

        // Pass 1: Create all phi nodes (without incoming values)
        for (block_id, mir_block) in &sorted_blocks {
            let llvm_block = self.block_map[block_id];
            self.builder.position_at_end(llvm_block);

            for phi in &mir_block.phi_nodes {
                self.create_phi_node(phi)?;
            }
        }

        // Pass 2: Compile all blocks (instructions and terminators)
        for (block_id, mir_block) in &sorted_blocks {
            let llvm_block = self.block_map[block_id];
            self.builder.position_at_end(llvm_block);

            // Compile instructions
            for instruction in &mir_block.instructions {
                self.compile_instruction(instruction, &function.register_types)
                    .map_err(|e| {
                        format!(
                            "In block {:?}, instruction {:?}: {}",
                            block_id, instruction, e
                        )
                    })?;
            }

            // Compile terminator (pass llvm_func for return type checking)
            self.compile_terminator(&mir_block.terminator, llvm_func)?;
        }

        // Pass 3: Fill in phi node incoming values
        for (block_id, mir_block) in &sorted_blocks {
            for phi in &mir_block.phi_nodes {
                self.fill_phi_incoming(phi)?;
            }
        }

        Ok(())
    }

    /// Create a phi node (without incoming values)
    fn create_phi_node(&mut self, phi: &IrPhiNode) -> Result<(), String> {
        // Skip void phi nodes - they have no value
        if phi.ty == IrType::Void {
            // Insert a placeholder for void phi
            let placeholder = self.context.i8_type().const_int(0, false).into();
            self.value_map.insert(phi.dest, placeholder);
            return Ok(());
        }

        let phi_ty = self.translate_type(&phi.ty)?;
        let llvm_phi = self
            .builder
            .build_phi(phi_ty, &format!("phi_{}", phi.dest.as_u32()))
            .map_err(|e| format!("Failed to build phi: {}", e))?;

        // Store the phi node for later filling
        self.phi_map.insert(phi.dest, llvm_phi);

        // Also add to value map so it can be used by instructions
        self.value_map.insert(phi.dest, llvm_phi.as_basic_value());
        Ok(())
    }

    /// Fill in phi node incoming values (after all blocks are compiled)
    fn fill_phi_incoming(&mut self, phi: &IrPhiNode) -> Result<(), String> {
        // Skip void phi nodes - they were given placeholders
        if phi.ty == IrType::Void {
            return Ok(());
        }

        let llvm_phi = self
            .phi_map
            .get(&phi.dest)
            .ok_or_else(|| format!("Phi node {:?} not found", phi.dest))?;

        let expected_ty = llvm_phi.as_basic_value().get_type();

        // Add incoming values
        for (block_id, value_id) in &phi.incoming {
            let llvm_block = self
                .block_map
                .get(block_id)
                .ok_or_else(|| format!("Block {:?} not found for phi", block_id))?;
            let llvm_value = self.value_map.get(value_id).ok_or_else(|| {
                format!(
                    "Value {:?} not found in value map for phi incoming",
                    value_id
                )
            })?;

            // Cast value to phi's expected type if there's a mismatch
            // This handles cases where MIR type tracking differs from actual computed types
            let actual_ty = llvm_value.get_type();
            let coerced_value = if actual_ty == expected_ty {
                *llvm_value
            } else {
                // Position builder BEFORE the terminator to insert cast
                let terminator = llvm_block.get_terminator();
                if let Some(term) = terminator {
                    self.builder.position_before(&term);
                } else {
                    self.builder.position_at_end(*llvm_block);
                }

                let cast_name = format!("phi_cast_{}", value_id.as_u32());

                // Handle int<->float conversions
                if llvm_value.is_float_value() && expected_ty.is_int_type() {
                    // Float to int: fptosi
                    self.builder
                        .build_float_to_signed_int(
                            llvm_value.into_float_value(),
                            expected_ty.into_int_type(),
                            &cast_name,
                        )
                        .map_err(|e| format!("Failed to cast phi float->int: {}", e))?
                        .into()
                } else if llvm_value.is_int_value() && expected_ty.is_float_type() {
                    // Int to float: sitofp
                    self.builder
                        .build_signed_int_to_float(
                            llvm_value.into_int_value(),
                            expected_ty.into_float_type(),
                            &cast_name,
                        )
                        .map_err(|e| format!("Failed to cast phi int->float: {}", e))?
                        .into()
                } else if llvm_value.is_int_value() && expected_ty.is_int_type() {
                    // Int to int: resize
                    let src_bits = llvm_value.into_int_value().get_type().get_bit_width();
                    let dst_bits = expected_ty.into_int_type().get_bit_width();
                    if src_bits < dst_bits {
                        self.builder
                            .build_int_z_extend(
                                llvm_value.into_int_value(),
                                expected_ty.into_int_type(),
                                &cast_name,
                            )
                            .map_err(|e| format!("Failed to extend phi int: {}", e))?
                            .into()
                    } else {
                        self.builder
                            .build_int_truncate(
                                llvm_value.into_int_value(),
                                expected_ty.into_int_type(),
                                &cast_name,
                            )
                            .map_err(|e| format!("Failed to truncate phi int: {}", e))?
                            .into()
                    }
                } else {
                    // For other cases, use as-is (might fail verification but gives better error)
                    *llvm_value
                }
            };

            llvm_phi.add_incoming(&[(&coerced_value, *llvm_block)]);
        }

        Ok(())
    }

    /// Compile a single MIR instruction to LLVM IR
    /// The register_types map is used to determine the proper types for operations
    fn compile_instruction(
        &mut self,
        inst: &IrInstruction,
        register_types: &HashMap<IrId, IrType>,
    ) -> Result<(), String> {
        match inst {
            IrInstruction::Const { dest, value } => {
                let llvm_value = self.compile_constant(value)?;
                self.value_map.insert(*dest, llvm_value);
            }

            IrInstruction::Copy { dest, src } => {
                let src_value = self.get_value(*src)?;
                self.value_map.insert(*dest, src_value);
            }

            IrInstruction::Load { dest, ptr, ty } => {
                // Skip void loads - insert placeholder
                if *ty == IrType::Void {
                    let placeholder = self.context.i8_type().const_int(0, false).into();
                    self.value_map.insert(*dest, placeholder);
                } else {
                    let ptr_value = self.get_value(*ptr)?;
                    // Handle case where pointer is stored as integer (from array element access)
                    let ptr = if ptr_value.is_pointer_value() {
                        ptr_value.into_pointer_value()
                    } else if ptr_value.is_int_value() {
                        self.builder
                            .build_int_to_ptr(
                                ptr_value.into_int_value(),
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                &format!("load_ptr_{}", ptr.as_u32()),
                            )
                            .map_err(|e| format!("Failed to convert int to ptr for load: {}", e))?
                    } else {
                        return Err(format!("Load ptr {:?} has unexpected type", ptr));
                    };
                    let load_ty = self.translate_type(ty)?;

                    let loaded = self
                        .builder
                        .build_load(load_ty, ptr, &format!("load_{}", dest.as_u32()))
                        .map_err(|e| format!("Failed to build load: {}", e))?;
                    self.value_map.insert(*dest, loaded);
                }
            }

            IrInstruction::Store { ptr, value } => {
                let ptr_raw = self.get_value(*ptr)?;
                // Handle case where pointer is stored as integer (from array element access)
                let ptr_val = if ptr_raw.is_pointer_value() {
                    ptr_raw.into_pointer_value()
                } else if ptr_raw.is_int_value() {
                    self.builder
                        .build_int_to_ptr(
                            ptr_raw.into_int_value(),
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            &format!("store_ptr_{}", ptr.as_u32()),
                        )
                        .map_err(|e| format!("Failed to convert int to ptr for store: {}", e))?
                } else {
                    return Err(format!("Store ptr {:?} has unexpected type", ptr));
                };
                let value_val = self.get_value(*value)?;
                self.builder
                    .build_store(ptr_val, value_val)
                    .map_err(|e| format!("Failed to build store: {}", e))?;
            }

            IrInstruction::BinOp {
                dest,
                op,
                left,
                right,
            } => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                // Get the result type from register_types - this tells us whether to use
                // integer or float operations, avoiding incorrect type inference
                let result_ty = register_types
                    .get(dest)
                    .or_else(|| register_types.get(left));
                let result = self.compile_binop(*op, left_val, right_val, *dest, result_ty)?;
                self.value_map.insert(*dest, result);
            }

            IrInstruction::UnOp { dest, op, operand } => {
                let operand_val = self.get_value(*operand)?;
                // Get the result type from register_types
                let result_ty = register_types
                    .get(dest)
                    .or_else(|| register_types.get(operand));
                let result = self.compile_unop(*op, operand_val, *dest, result_ty)?;
                self.value_map.insert(*dest, result);
            }

            IrInstruction::Cmp {
                dest,
                op,
                left,
                right,
            } => {
                let left_val = self.get_value(*left)?;
                let right_val = self.get_value(*right)?;
                // Get operand type for comparison (comparison result is always Bool)
                let operand_ty = register_types.get(left);
                let result = self.compile_compare(*op, left_val, right_val, *dest, operand_ty)?;
                self.value_map.insert(*dest, result);
            }

            IrInstruction::CallDirect {
                dest,
                func_id,
                args,
                arg_ownership: _,
                type_args: _,
                is_tail_call: _,
            } => {
                // Note: type_args are handled by monomorphization pass before codegen
                let result = self.compile_direct_call(*func_id, args)?;
                if let Some(dest) = dest {
                    if let Some(result_val) = result {
                        self.value_map.insert(*dest, result_val);
                    } else {
                        // Void function but dest expected - insert placeholder
                        let placeholder = self.context.i8_type().const_int(0, false).into();
                        self.value_map.insert(*dest, placeholder);
                    }
                }
            }

            IrInstruction::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => {
                let cond_raw = self.get_value(*condition)?;
                // Condition must be i1 (boolean) - convert float to bool via comparison with 0.0
                let cond = if cond_raw.is_float_value() {
                    self.builder
                        .build_float_compare(
                            inkwell::FloatPredicate::ONE,
                            cond_raw.into_float_value(),
                            cond_raw.get_type().into_float_type().const_zero(),
                            "select_cond_cast",
                        )
                        .map_err(|e| format!("Failed to cast select condition: {}", e))?
                } else {
                    cond_raw.into_int_value()
                };
                let true_v = self.get_value(*true_val)?;
                let false_v = self.get_value(*false_val)?;

                let result = self
                    .builder
                    .build_select(cond, true_v, false_v, &format!("select_{}", dest.as_u32()))
                    .map_err(|e| format!("Failed to build select: {}", e))?;
                self.value_map.insert(*dest, result);
            }

            IrInstruction::Alloc { dest, ty, count } => {
                let alloc_ty = self.translate_type(ty)?;

                // Use alloca (stack allocation) for fixed-size allocations without a
                // dynamic count. The MIR Free instructions become no-ops. This eliminates
                // malloc/free overhead in hot loops (89% of mandelbrot time was in allocator).
                if count.is_none() {
                    self.alloca_ids.insert(*dest);

                    // WORKAROUND: For Ptr types that might represent HaxeArray or other
                    // runtime structs, allocate extra space (128 bytes = 8 * 16 elements).
                    // This matches Cranelift's behavior for Ptr/Any types.
                    // HaxeArray is 32 bytes: { ptr, len, cap, elem_size }
                    let final_alloc_ty: BasicTypeEnum = match ty {
                        IrType::Ptr(_) | IrType::Any | IrType::Ref(_) => {
                            // Allocate 128 bytes as an i8 array (matches Cranelift's 8*16)
                            self.context.i8_type().array_type(128).into()
                        }
                        _ => alloc_ty,
                    };

                    let alloca = self
                        .builder
                        .build_alloca(final_alloc_ty, &format!("stack_{}", dest.as_u32()))
                        .map_err(|e| format!("Failed to build alloca: {}", e))?;
                    self.value_map.insert(*dest, alloca.into());
                } else {
                    // JIT mode or dynamic-count: use malloc() for heap allocation.
                    // The MIR layer emits Free instructions that call C free(), so we must
                    // allocate via malloc to match. Using alloca would crash when free() is
                    // called on stack pointers.

                    // Get element size - use 8 bytes as default for unknown types
                    let element_size = if let Some(size_val) = alloc_ty.size_of() {
                        self.builder
                            .build_int_z_extend_or_bit_cast(
                                size_val,
                                self.context.i64_type(),
                                "elem_size",
                            )
                            .map_err(|e| format!("Failed to cast element size: {}", e))?
                    } else {
                        self.context.i64_type().const_int(8, false)
                    };

                    let total_size = if let Some(count_id) = count {
                        let count_raw = self.get_value(*count_id)?;
                        let count_val = if count_raw.is_float_value() {
                            self.builder
                                .build_float_to_signed_int(
                                    count_raw.into_float_value(),
                                    self.context.i64_type(),
                                    "alloc_count_cast",
                                )
                                .map_err(|e| format!("Failed to cast alloc count: {}", e))?
                        } else {
                            let raw_int = count_raw.into_int_value();
                            if raw_int.get_type().get_bit_width() < 64 {
                                self.builder
                                    .build_int_z_extend(
                                        raw_int,
                                        self.context.i64_type(),
                                        "count_ext",
                                    )
                                    .map_err(|e| format!("Failed to extend count: {}", e))?
                            } else {
                                raw_int
                            }
                        };
                        // total_size = element_size * count
                        self.builder
                            .build_int_mul(element_size, count_val, "total_size")
                            .map_err(|e| format!("Failed to compute total size: {}", e))?
                    } else {
                        element_size
                    };

                    // Get or declare malloc function
                    let malloc_fn = match self.module.get_function("malloc") {
                        Some(f) => f,
                        None => {
                            let malloc_fn_type = self
                                .context
                                .ptr_type(AddressSpace::default())
                                .fn_type(&[self.context.i64_type().into()], false);
                            self.module.add_function("malloc", malloc_fn_type, None)
                        }
                    };

                    // Call malloc(total_size)
                    let malloc_result = self
                        .builder
                        .build_call(
                            malloc_fn,
                            &[total_size.into()],
                            &format!("malloc_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to build malloc call: {}", e))?;

                    let ptr = malloc_result
                        .try_as_basic_value()
                        .left()
                        .ok_or("malloc did not return a value")?
                        .into_pointer_value();

                    self.value_map.insert(*dest, ptr.into());
                }
            }

            IrInstruction::GetElementPtr {
                dest,
                ptr,
                indices,
                ty,
            } => {
                // Get the pointer value - may be an actual pointer or an integer (from array element load)
                let raw_val = self.get_value(*ptr)?;
                let ptr_val = if raw_val.is_pointer_value() {
                    raw_val.into_pointer_value()
                } else if raw_val.is_int_value() {
                    // Convert integer to pointer (e.g., Body pointer loaded as I64 from array)
                    let int_val = raw_val.into_int_value();
                    self.builder
                        .build_int_to_ptr(
                            int_val,
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            &format!("int_to_ptr_{}", ptr.as_u32()),
                        )
                        .map_err(|e| format!("Failed to convert int to ptr for GEP: {}", e))?
                } else {
                    return Err(format!(
                        "GEP ptr {:?} has unexpected type (not pointer or int)",
                        ptr
                    ));
                };

                // Calculate element size from the pointer type.
                // MIR GEP can use either field indices (multiplied by 8) or byte offsets
                // (for Ptr(I8) types where index is already a byte offset).
                // We extract the pointee type and compute its size.
                let elem_size: u64 = match ty {
                    crate::ir::IrType::Ptr(inner) | crate::ir::IrType::Ref(inner) => {
                        Self::ir_type_size(inner)
                    }
                    _ => 8, // Default to 8-byte fields for non-pointer types
                };

                let mut index_vals = Vec::new();
                for &id in indices {
                    let val = self.get_value(id)?;
                    let int_val = if val.is_float_value() {
                        // Convert float to i64
                        self.builder
                            .build_float_to_signed_int(
                                val.into_float_value(),
                                self.context.i64_type(),
                                "gep_idx_cast",
                            )
                            .map_err(|e| format!("Failed to cast GEP index: {}", e))?
                    } else {
                        // Extend to i64 if needed for consistent multiplication
                        let raw_int = val.into_int_value();
                        if raw_int.get_type().get_bit_width() < 64 {
                            self.builder
                                .build_int_s_extend(raw_int, self.context.i64_type(), "gep_idx_ext")
                                .map_err(|e| format!("Failed to extend GEP index: {}", e))?
                        } else {
                            raw_int
                        }
                    };

                    // Multiply index by element size to get byte offset
                    // For Ptr(I8), elem_size=1 so index is used directly as byte offset
                    // For Ptr(I64), elem_size=8 so index is multiplied by 8
                    let byte_offset = if elem_size == 1 {
                        // Optimization: skip multiplication for byte-addressed GEPs
                        int_val
                    } else {
                        self.builder
                            .build_int_mul(
                                int_val,
                                self.context.i64_type().const_int(elem_size, false),
                                "field_byte_offset",
                            )
                            .map_err(|e| format!("Failed to multiply field index: {}", e))?
                    };

                    index_vals.push(byte_offset);
                }

                unsafe {
                    let gep = self
                        .builder
                        .build_gep(
                            self.context.i8_type(),
                            ptr_val,
                            &index_vals,
                            &format!("gep_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to build GEP: {}", e))?;
                    self.value_map.insert(*dest, gep.into());
                }
            }

            IrInstruction::Cast {
                dest,
                src,
                from_ty,
                to_ty,
            } => {
                let src_val = self.get_value(*src)?;
                let result = self.compile_cast(src_val, from_ty, to_ty, *dest)?;
                self.value_map.insert(*dest, result);
            }

            IrInstruction::BitCast { dest, src, ty } => {
                let src_val = self.get_value(*src)?;
                let target_ty = self.translate_type(ty)?;

                let result = if src_val.is_int_value() {
                    // If target is pointer type, use inttoptr instead of bitcast
                    // (LLVM bitcast cannot convert integer to ptr)
                    if target_ty.is_pointer_type() {
                        self.builder
                            .build_int_to_ptr(
                                src_val.into_int_value(),
                                target_ty.into_pointer_type(),
                                &format!("bitcast_{}", dest.as_u32()),
                            )
                            .map(|v| v.into())
                    } else {
                        self.builder.build_bit_cast(
                            src_val.into_int_value(),
                            target_ty,
                            &format!("bitcast_{}", dest.as_u32()),
                        )
                    }
                } else if src_val.is_float_value() {
                    self.builder.build_bit_cast(
                        src_val.into_float_value(),
                        target_ty,
                        &format!("bitcast_{}", dest.as_u32()),
                    )
                } else if src_val.is_pointer_value() {
                    // If target is integer type, use ptrtoint instead of bitcast
                    // (LLVM bitcast cannot convert ptr to integer)
                    if target_ty.is_int_type() {
                        self.builder
                            .build_ptr_to_int(
                                src_val.into_pointer_value(),
                                target_ty.into_int_type(),
                                &format!("bitcast_{}", dest.as_u32()),
                            )
                            .map(|v| v.into())
                    } else {
                        self.builder.build_bit_cast(
                            src_val.into_pointer_value(),
                            target_ty,
                            &format!("bitcast_{}", dest.as_u32()),
                        )
                    }
                } else {
                    return Err("Unsupported bitcast type".to_string());
                }
                .map_err(|e| format!("Failed to build bitcast: {}", e))?;

                self.value_map.insert(*dest, result);
            }

            IrInstruction::CallIndirect {
                dest,
                func_ptr,
                args,
                signature,
                arg_ownership: _,
                is_tail_call: _,
            } => {
                // func_ptr is a pointer to a closure struct: { fn_ptr: i64, env_ptr: i64 }
                let closure_ptr = self.get_value(*func_ptr)?.into_pointer_value();
                let i64_type = self.context.i64_type();
                let ptr_type = self.context.ptr_type(AddressSpace::default());

                // Load function pointer from closure offset 0
                let fn_ptr_i64 = self
                    .builder
                    .build_load(i64_type, closure_ptr, "cl_fn_ptr")
                    .map_err(|e| format!("Failed to load closure fn_ptr: {}", e))?
                    .into_int_value();
                let fn_ptr_val = self
                    .builder
                    .build_int_to_ptr(fn_ptr_i64, ptr_type, "cl_fn_ptr_as_ptr")
                    .map_err(|e| format!("Failed to inttoptr fn_ptr: {}", e))?;

                // Load environment pointer from closure offset 8
                let env_slot = unsafe {
                    self.builder
                        .build_gep(
                            self.context.i8_type(),
                            closure_ptr,
                            &[i64_type.const_int(8, false)],
                            "cl_env_slot",
                        )
                        .map_err(|e| format!("Failed to GEP closure env: {}", e))?
                };
                let env_ptr_i64 = self
                    .builder
                    .build_load(i64_type, env_slot, "cl_env_ptr")
                    .map_err(|e| format!("Failed to load closure env_ptr: {}", e))?;

                // Build argument list: env_ptr first, then user args
                let mut call_args: Vec<BasicMetadataValueEnum> = Vec::new();
                call_args.push(env_ptr_i64.into());
                for &arg_id in args.iter() {
                    let arg_val = self.get_value(arg_id)?;
                    call_args.push(arg_val.into());
                }

                // Build function type with env_ptr prepended
                let fn_type = {
                    match signature {
                        IrType::Function {
                            params,
                            return_type,
                            varargs,
                        } => {
                            // Build param types: env_ptr (i64) + user params
                            let mut param_types: Vec<BasicMetadataTypeEnum> =
                                Vec::with_capacity(params.len() + 1);
                            param_types.push(i64_type.into()); // env_ptr
                            for p in params {
                                let llvm_ty = self.translate_type(p)?;
                                param_types.push(llvm_ty.into());
                            }

                            if matches!(return_type.as_ref(), IrType::Void) {
                                self.context.void_type().fn_type(&param_types, *varargs)
                            } else {
                                let ret_ty = self.translate_type(return_type)?;
                                ret_ty.fn_type(&param_types, *varargs)
                            }
                        }
                        _ => {
                            return Err(format!(
                                "Invalid signature type for CallIndirect: {:?}",
                                signature
                            ))
                        }
                    }
                };

                let call_site = self
                    .builder
                    .build_indirect_call(fn_type, fn_ptr_val, &call_args, "indirect_call")
                    .map_err(|e| format!("Failed to build indirect call: {}", e))?;

                if let Some(dest) = dest {
                    if let Some(result_val) = call_site.try_as_basic_value().left() {
                        self.value_map.insert(*dest, result_val);
                    } else {
                        let placeholder = self.context.i8_type().const_int(0, false).into();
                        self.value_map.insert(*dest, placeholder);
                    }
                }
            }

            IrInstruction::Free { ptr } => {
                // If the pointer came from alloca (stack), Free is a no-op.
                // Otherwise, call libc free() for heap-allocated memory.
                if !self.alloca_ids.contains(ptr) {
                    if let Ok(ptr_val) = self.get_value(*ptr) {
                        // Get or declare free function
                        let free_fn = match self.module.get_function("free") {
                            Some(f) => f,
                            None => {
                                let free_fn_type = self.context.void_type().fn_type(
                                    &[self.context.ptr_type(Default::default()).into()],
                                    false,
                                );
                                self.module.add_function("free", free_fn_type, None)
                            }
                        };

                        // Cast to pointer if needed and call free(ptr)
                        let ptr_as_ptr = if ptr_val.is_pointer_value() {
                            ptr_val.into_pointer_value()
                        } else {
                            // int-to-ptr cast for non-pointer values
                            self.builder
                                .build_int_to_ptr(
                                    ptr_val.into_int_value(),
                                    self.context.ptr_type(Default::default()),
                                    &format!("free_cast_{}", ptr.as_u32()),
                                )
                                .map_err(|e| format!("Failed to cast for free: {}", e))?
                        };

                        let _ = self.builder.build_call(
                            free_fn,
                            &[ptr_as_ptr.into()],
                            &format!("free_{}", ptr.as_u32()),
                        );
                    }
                }
            }

            IrInstruction::MemCopy { dest, src, size } => {
                let dest_ptr = self.get_value(*dest)?.into_pointer_value();
                let src_ptr = self.get_value(*src)?.into_pointer_value();
                let size_raw = self.get_value(*size)?;
                let size_val = if size_raw.is_float_value() {
                    self.builder
                        .build_float_to_unsigned_int(
                            size_raw.into_float_value(),
                            self.context.i64_type(),
                            "memcpy_size_cast",
                        )
                        .map_err(|e| format!("Failed to cast memcpy size: {}", e))?
                } else {
                    size_raw.into_int_value()
                };

                // Use LLVM's memcpy intrinsic with default alignment (1 byte for i8*)
                self.builder
                    .build_memcpy(
                        dest_ptr, 1, // alignment for i8* (can be optimized by LLVM)
                        src_ptr, 1, // alignment
                        size_val,
                    )
                    .map_err(|e| format!("Failed to build memcpy: {}", e))?;
            }

            IrInstruction::MemSet { dest, value, size } => {
                let dest_ptr = self.get_value(*dest)?.into_pointer_value();
                let value_raw = self.get_value(*value)?;
                let value_val = if value_raw.is_float_value() {
                    self.builder
                        .build_float_to_unsigned_int(
                            value_raw.into_float_value(),
                            self.context.i8_type(),
                            "memset_val_cast",
                        )
                        .map_err(|e| format!("Failed to cast memset value: {}", e))?
                } else {
                    value_raw.into_int_value()
                };
                let size_raw = self.get_value(*size)?;
                let size_val = if size_raw.is_float_value() {
                    self.builder
                        .build_float_to_unsigned_int(
                            size_raw.into_float_value(),
                            self.context.i64_type(),
                            "memset_size_cast",
                        )
                        .map_err(|e| format!("Failed to cast memset size: {}", e))?
                } else {
                    size_raw.into_int_value()
                };

                // Use LLVM's memset intrinsic with default alignment
                self.builder
                    .build_memset(
                        dest_ptr, 1, // alignment for i8* (can be optimized by LLVM)
                        value_val, size_val,
                    )
                    .map_err(|e| format!("Failed to build memset: {}", e))?;
            }
            IrInstruction::Throw { .. } => {
                return Err("Throw not yet implemented".to_string());
            }
            IrInstruction::LandingPad { .. } => {
                return Err("LandingPad not yet implemented".to_string());
            }
            IrInstruction::Resume { .. } => {
                return Err("Resume not yet implemented".to_string());
            }
            IrInstruction::ExtractValue {
                dest,
                aggregate,
                indices,
            } => {
                let agg_val = self.get_value(*aggregate)?;

                let result = if agg_val.is_struct_value() {
                    self.builder
                        .build_extract_value(
                            agg_val.into_struct_value(),
                            indices[0],
                            &format!("extract_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to build extract_value: {}", e))?
                } else if agg_val.is_array_value() {
                    self.builder
                        .build_extract_value(
                            agg_val.into_array_value(),
                            indices[0],
                            &format!("extract_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to build extract_value: {}", e))?
                } else {
                    return Err("ExtractValue only works on struct or array values".to_string());
                };

                self.value_map.insert(*dest, result);
            }

            IrInstruction::InsertValue {
                dest,
                aggregate,
                value,
                indices,
            } => {
                let agg_val = self.get_value(*aggregate)?;
                let insert_val = self.get_value(*value)?;

                let result = if agg_val.is_struct_value() {
                    let struct_val = self
                        .builder
                        .build_insert_value(
                            agg_val.into_struct_value(),
                            insert_val,
                            indices[0],
                            &format!("insert_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to build insert_value: {}", e))?;
                    struct_val.as_basic_value_enum()
                } else if agg_val.is_array_value() {
                    let array_val = self
                        .builder
                        .build_insert_value(
                            agg_val.into_array_value(),
                            insert_val,
                            indices[0],
                            &format!("insert_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to build insert_value: {}", e))?;
                    array_val.as_basic_value_enum()
                } else {
                    return Err("InsertValue only works on struct or array values".to_string());
                };

                self.value_map.insert(*dest, result);
            }
            IrInstruction::MakeClosure {
                dest,
                func_id,
                captured_values,
            } => {
                // Get function pointer for the lambda
                let llvm_func = self.function_map.get(func_id).ok_or_else(|| {
                    format!("Lambda function {:?} not found in function_map", func_id)
                })?;
                let func_addr = llvm_func.as_global_value().as_pointer_value();

                let ptr_type = self.context.ptr_type(AddressSpace::default());
                let i64_type = self.context.i64_type();

                // Allocate environment for captured values (if any)
                let env_ptr = if !captured_values.is_empty() {
                    let env_size = (captured_values.len() * 8) as u64;

                    // Get or declare malloc
                    let malloc_fn = match self.module.get_function("malloc") {
                        Some(f) => f,
                        None => {
                            let malloc_fn_type = ptr_type.fn_type(&[i64_type.into()], false);
                            self.module.add_function("malloc", malloc_fn_type, None)
                        }
                    };

                    let size_arg = i64_type.const_int(env_size, false);
                    let env_addr = self
                        .builder
                        .build_call(malloc_fn, &[size_arg.into()], "env_malloc")
                        .map_err(|e| format!("Failed to malloc env: {}", e))?
                        .try_as_basic_value()
                        .left()
                        .ok_or("malloc did not return a value")?
                        .into_pointer_value();

                    // Store each captured value into the environment
                    for (i, captured_id) in captured_values.iter().enumerate() {
                        let captured_val = self.get_value(*captured_id)?;
                        let offset = (i * 8) as u64;

                        // All env slots are i64 — extend smaller ints
                        let val_as_i64 = if captured_val.is_int_value() {
                            let int_val = captured_val.into_int_value();
                            if int_val.get_type().get_bit_width() < 64 {
                                self.builder
                                    .build_int_s_extend(int_val, i64_type, "env_extend")
                                    .map_err(|e| format!("Failed to extend env val: {}", e))?
                                    .into()
                            } else {
                                int_val.into()
                            }
                        } else {
                            // For pointers/floats, bitcast to i64
                            self.builder
                                .build_bit_cast(captured_val, i64_type, "env_cast")
                                .map_err(|e| format!("Failed to bitcast env val: {}", e))?
                        };

                        let slot_ptr = if offset == 0 {
                            env_addr
                        } else {
                            unsafe {
                                self.builder
                                    .build_gep(
                                        self.context.i8_type(),
                                        env_addr,
                                        &[i64_type.const_int(offset, false)],
                                        &format!("env_slot_{}", i),
                                    )
                                    .map_err(|e| format!("Failed to GEP env slot: {}", e))?
                            }
                        };

                        self.builder
                            .build_store(slot_ptr, val_as_i64)
                            .map_err(|e| format!("Failed to store env val: {}", e))?;
                    }

                    env_addr
                } else {
                    // No captures - null environment pointer
                    ptr_type.const_null()
                };

                // Allocate closure struct: { fn_ptr: ptr, env_ptr: ptr }
                let malloc_fn = match self.module.get_function("malloc") {
                    Some(f) => f,
                    None => {
                        let malloc_fn_type = ptr_type.fn_type(&[i64_type.into()], false);
                        self.module.add_function("malloc", malloc_fn_type, None)
                    }
                };

                let closure_size = i64_type.const_int(16, false);
                let closure_ptr = self
                    .builder
                    .build_call(malloc_fn, &[closure_size.into()], "closure_malloc")
                    .map_err(|e| format!("Failed to malloc closure: {}", e))?
                    .try_as_basic_value()
                    .left()
                    .ok_or("malloc did not return a value")?
                    .into_pointer_value();

                // Store function pointer at offset 0
                let func_as_ptr = self
                    .builder
                    .build_ptr_to_int(func_addr, i64_type, "func_as_i64")
                    .map_err(|e| format!("Failed to ptrtoint func: {}", e))?;
                self.builder
                    .build_store(closure_ptr, func_as_ptr)
                    .map_err(|e| format!("Failed to store fn_ptr: {}", e))?;

                // Store environment pointer at offset 8
                let env_slot = unsafe {
                    self.builder
                        .build_gep(
                            self.context.i8_type(),
                            closure_ptr,
                            &[i64_type.const_int(8, false)],
                            "closure_env_slot",
                        )
                        .map_err(|e| format!("Failed to GEP env slot: {}", e))?
                };
                let env_as_i64 = self
                    .builder
                    .build_ptr_to_int(env_ptr, i64_type, "env_as_i64")
                    .map_err(|e| format!("Failed to ptrtoint env: {}", e))?;
                self.builder
                    .build_store(env_slot, env_as_i64)
                    .map_err(|e| format!("Failed to store env_ptr: {}", e))?;

                self.value_map.insert(*dest, closure_ptr.into());
            }
            IrInstruction::ClosureFunc { dest, closure } => {
                // Load function pointer from closure offset 0
                let closure_val = self.get_value(*closure)?.into_pointer_value();
                let func_ptr_val = self
                    .builder
                    .build_load(self.context.i64_type(), closure_val, "closure_func")
                    .map_err(|e| format!("Failed to load closure func: {}", e))?;
                self.value_map.insert(*dest, func_ptr_val);
            }
            IrInstruction::ClosureEnv { dest, closure } => {
                // Load environment pointer from closure offset 8
                let closure_val = self.get_value(*closure)?.into_pointer_value();
                let i64_type = self.context.i64_type();
                let env_slot = unsafe {
                    self.builder
                        .build_gep(
                            self.context.i8_type(),
                            closure_val,
                            &[i64_type.const_int(8, false)],
                            "closure_env_gep",
                        )
                        .map_err(|e| format!("Failed to GEP closure env: {}", e))?
                };
                let env_val = self
                    .builder
                    .build_load(i64_type, env_slot, "closure_env")
                    .map_err(|e| format!("Failed to load closure env: {}", e))?;
                // Convert i64 back to pointer
                let env_ptr = self
                    .builder
                    .build_int_to_ptr(
                        env_val.into_int_value(),
                        self.context.ptr_type(AddressSpace::default()),
                        "env_ptr",
                    )
                    .map_err(|e| format!("Failed to inttoptr env: {}", e))?;
                self.value_map.insert(*dest, env_ptr.into());
            }
            IrInstruction::DebugLoc { .. } => {
                // Debug locations are metadata, skip for now
            }
            IrInstruction::InlineAsm { .. } => {
                return Err("InlineAsm not yet implemented".to_string());
            }

            // Control flow is handled by terminators, not regular instructions
            IrInstruction::Jump { .. }
            | IrInstruction::Branch { .. }
            | IrInstruction::Switch { .. }
            | IrInstruction::Return { .. } => {
                return Err("Control flow instructions should be terminators".to_string());
            }

            IrInstruction::Phi { .. } => {
                return Err("Phi nodes should be in phi_nodes list".to_string());
            }

            // Ownership/memory instructions - treat as copies for now
            IrInstruction::Move { dest, src } => {
                // Move is a copy in LLVM (ownership is a Rust concept)
                let val = self.get_value(*src)?;
                self.value_map.insert(*dest, val);
            }
            IrInstruction::BorrowImmutable { dest, src, .. } => {
                // Borrow is a pointer in LLVM
                let val = self.get_value(*src)?;
                self.value_map.insert(*dest, val);
            }
            IrInstruction::BorrowMutable { dest, src, .. } => {
                // Mutable borrow is also a pointer
                let val = self.get_value(*src)?;
                self.value_map.insert(*dest, val);
            }
            IrInstruction::EndBorrow { .. } => {
                // End borrow is a no-op in LLVM (ownership system handles cleanup)
            }
            IrInstruction::Clone { dest, src } => {
                // Clone is a copy in LLVM
                let val = self.get_value(*src)?;
                self.value_map.insert(*dest, val);
            }
            IrInstruction::Copy { dest, src } => {
                let val = self.get_value(*src)?;
                self.value_map.insert(*dest, val);
            }
            IrInstruction::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => {
                let cond_raw = self.get_value(*condition)?;
                // Condition must be i1 (boolean) - convert float to bool via comparison with 0.0
                let cond_val = if cond_raw.is_float_value() {
                    self.builder
                        .build_float_compare(
                            inkwell::FloatPredicate::ONE,
                            cond_raw.into_float_value(),
                            cond_raw.get_type().into_float_type().const_zero(),
                            "select_cond_cast",
                        )
                        .map_err(|e| format!("Failed to cast select condition: {}", e))?
                } else {
                    cond_raw.into_int_value()
                };
                let true_v = self.get_value(*true_val)?;
                let false_v = self.get_value(*false_val)?;
                let result = self
                    .builder
                    .build_select(
                        cond_val,
                        true_v,
                        false_v,
                        &format!("select_{}", dest.as_u32()),
                    )
                    .map_err(|e| format!("Failed to build select: {}", e))?;
                self.value_map.insert(*dest, result);
            }

            // Union operations
            // Union layout: {i32 tag, [i8 x data_size]}
            IrInstruction::CreateUnion {
                dest,
                discriminant,
                value,
                ty,
            } => {
                let union_ty = self.translate_type(ty)?;
                let union_struct_ty = union_ty.into_struct_type();

                // Allocate union on stack
                let union_ptr = self
                    .builder
                    .build_alloca(union_struct_ty, &format!("union_alloca_{}", dest.as_u32()))
                    .map_err(|e| format!("Failed to alloca union: {}", e))?;

                // Store tag (discriminant) at field 0
                let tag_ptr = self
                    .builder
                    .build_struct_gep(union_struct_ty, union_ptr, 0, "tag_ptr")
                    .map_err(|e| format!("Failed to get tag ptr: {}", e))?;
                let tag_val = self
                    .context
                    .i32_type()
                    .const_int(*discriminant as u64, false);
                self.builder
                    .build_store(tag_ptr, tag_val)
                    .map_err(|e| format!("Failed to store tag: {}", e))?;

                // Store value in data area (field 1)
                let data_ptr = self
                    .builder
                    .build_struct_gep(union_struct_ty, union_ptr, 1, "data_ptr")
                    .map_err(|e| format!("Failed to get data ptr: {}", e))?;

                // Get the value and store it via pointer cast
                let value_val = self.get_value(*value)?;
                let value_ptr = self
                    .builder
                    .build_alloca(value_val.get_type(), "value_tmp")
                    .map_err(|e| format!("Failed to alloca value: {}", e))?;
                self.builder
                    .build_store(value_ptr, value_val)
                    .map_err(|e| format!("Failed to store value: {}", e))?;

                // Memcpy value to data area
                let i8_ptr_ty = self.context.ptr_type(AddressSpace::default());
                let data_ptr_cast = self
                    .builder
                    .build_pointer_cast(data_ptr, i8_ptr_ty, "data_ptr_cast")
                    .map_err(|e| format!("Failed to cast data ptr: {}", e))?;
                let value_ptr_cast = self
                    .builder
                    .build_pointer_cast(value_ptr, i8_ptr_ty, "value_ptr_cast")
                    .map_err(|e| format!("Failed to cast value ptr: {}", e))?;

                // Get size of value type
                let value_size = if let Some(ref td) = self.target_data {
                    td.get_store_size(&value_val.get_type())
                } else {
                    8 // Default
                };
                let size_val = self.context.i64_type().const_int(value_size, false);

                // Build memcpy intrinsic call
                self.builder
                    .build_memcpy(data_ptr_cast, 1, value_ptr_cast, 1, size_val)
                    .map_err(|e| format!("Failed to build memcpy: {}", e))?;

                // Load the complete union value
                let union_val = self
                    .builder
                    .build_load(
                        union_struct_ty,
                        union_ptr,
                        &format!("union_{}", dest.as_u32()),
                    )
                    .map_err(|e| format!("Failed to load union: {}", e))?;
                self.value_map.insert(*dest, union_val);
            }

            IrInstruction::ExtractDiscriminant { dest, union_val } => {
                let union_value = self.get_value(*union_val)?;

                // Extract tag from field 0
                let tag = self
                    .builder
                    .build_extract_value(
                        union_value.into_struct_value(),
                        0,
                        &format!("tag_{}", dest.as_u32()),
                    )
                    .map_err(|e| format!("Failed to extract tag: {}", e))?;
                self.value_map.insert(*dest, tag);
            }

            IrInstruction::ExtractUnionValue {
                dest,
                union_val,
                value_ty,
                ..
            } => {
                let union_value = self.get_value(*union_val)?;
                let target_ty = self.translate_type(value_ty)?;

                // Get the union struct type for GEP
                let union_struct_ty = union_value.get_type().into_struct_type();

                // Allocate union on stack to get at data
                let union_ptr = self
                    .builder
                    .build_alloca(union_struct_ty, "union_extract_tmp")
                    .map_err(|e| format!("Failed to alloca for extract: {}", e))?;
                self.builder
                    .build_store(union_ptr, union_value)
                    .map_err(|e| format!("Failed to store union for extract: {}", e))?;

                // Get data pointer (field 1)
                let data_ptr = self
                    .builder
                    .build_struct_gep(union_struct_ty, union_ptr, 1, "data_ptr")
                    .map_err(|e| format!("Failed to get data ptr for extract: {}", e))?;

                // Cast data pointer to target type pointer and load
                let target_ptr_ty = self.context.ptr_type(AddressSpace::default());
                let typed_ptr = self
                    .builder
                    .build_pointer_cast(data_ptr, target_ptr_ty, "typed_data_ptr")
                    .map_err(|e| format!("Failed to cast data ptr: {}", e))?;

                let extracted = self
                    .builder
                    .build_load(
                        target_ty,
                        typed_ptr,
                        &format!("extracted_{}", dest.as_u32()),
                    )
                    .map_err(|e| format!("Failed to load extracted value: {}", e))?;
                self.value_map.insert(*dest, extracted);
            }

            // Struct operations
            IrInstruction::CreateStruct { dest, ty, fields } => {
                let struct_ty = self.translate_type(ty)?;

                // Start with an undef struct value
                let mut struct_val = struct_ty.into_struct_type().get_undef();

                // Insert each field value
                for (i, field_id) in fields.iter().enumerate() {
                    let field_val = self.get_value(*field_id)?;
                    struct_val = self
                        .builder
                        .build_insert_value(
                            struct_val,
                            field_val,
                            i as u32,
                            &format!("struct_field_{}", i),
                        )
                        .map_err(|e| format!("Failed to insert struct field: {}", e))?
                        .into_struct_value();
                }

                self.value_map.insert(*dest, struct_val.into());
            }

            // Pointer operations
            IrInstruction::PtrAdd {
                dest, ptr, offset, ..
            } => {
                let ptr_val = self.get_value(*ptr)?.into_pointer_value();
                let offset_raw = self.get_value(*offset)?;
                let offset_val = if offset_raw.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            offset_raw.into_float_value(),
                            self.context.i64_type(),
                            "ptr_offset_cast",
                        )
                        .map_err(|e| format!("Failed to cast ptr offset: {}", e))?
                } else {
                    offset_raw.into_int_value()
                };
                let result = unsafe {
                    self.builder
                        .build_gep(
                            self.context.i8_type(),
                            ptr_val,
                            &[offset_val],
                            &format!("ptradj_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to build PtrAdd: {}", e))?
                };
                self.value_map.insert(*dest, result.into());
            }

            // Special values
            IrInstruction::Undef { dest, ty } => {
                // Handle void undef with placeholder
                if *ty == IrType::Void {
                    let placeholder = self.context.i8_type().const_int(0, false).into();
                    self.value_map.insert(*dest, placeholder);
                } else {
                    let llvm_ty = self.translate_type(ty)?;
                    let undef_val = llvm_ty.const_zero(); // Use zero as placeholder for undef
                    self.value_map.insert(*dest, undef_val);
                }
            }

            // Function reference
            IrInstruction::FunctionRef { dest, func_id } => {
                // All functions should be declared before bodies are compiled
                if let Some(llvm_func) = self.function_map.get(func_id) {
                    let ptr = llvm_func.as_global_value().as_pointer_value();
                    self.value_map.insert(*dest, ptr.into());
                } else {
                    return Err(format!("Function {:?} not found in function_map for FunctionRef. Ensure all modules are compiled in declare-first order.", func_id));
                }
            }

            // === SIMD Vector Operations ===
            IrInstruction::VectorLoad { dest, ptr, vec_ty } => {
                let raw_ptr = self.get_value(*ptr)?;
                let ptr_val = if raw_ptr.is_pointer_value() {
                    raw_ptr.into_pointer_value()
                } else if raw_ptr.is_int_value() {
                    self.builder
                        .build_int_to_ptr(
                            raw_ptr.into_int_value(),
                            self.context.ptr_type(AddressSpace::default()),
                            &format!("vload_ptr_{}", ptr.as_u32()),
                        )
                        .map_err(|e| {
                            format!("Failed to convert int to ptr for vector load: {}", e)
                        })?
                } else {
                    return Err(format!("VectorLoad ptr {:?} has unexpected type", ptr));
                };
                let vec_llvm_ty = self.translate_type(vec_ty)?;
                let loaded = self
                    .builder
                    .build_load(vec_llvm_ty, ptr_val, &format!("vload_{}", dest.as_u32()))
                    .map_err(|e| format!("Failed to build vector load: {}", e))?;
                self.value_map.insert(*dest, loaded);
            }

            IrInstruction::VectorStore {
                ptr,
                value,
                vec_ty: _,
            } => {
                let raw_ptr = self.get_value(*ptr)?;
                let ptr_val = if raw_ptr.is_pointer_value() {
                    raw_ptr.into_pointer_value()
                } else if raw_ptr.is_int_value() {
                    self.builder
                        .build_int_to_ptr(
                            raw_ptr.into_int_value(),
                            self.context.ptr_type(AddressSpace::default()),
                            &format!("vstore_ptr_{}", ptr.as_u32()),
                        )
                        .map_err(|e| {
                            format!("Failed to convert int to ptr for vector store: {}", e)
                        })?
                } else {
                    return Err(format!("VectorStore ptr {:?} has unexpected type", ptr));
                };
                let vec_val = self.get_value(*value)?;
                self.builder
                    .build_store(ptr_val, vec_val)
                    .map_err(|e| format!("Failed to build vector store: {}", e))?;
            }

            IrInstruction::VectorBinOp {
                dest,
                op,
                left,
                right,
                vec_ty,
            } => {
                let lhs = self.get_value(*left)?;
                let rhs = self.get_value(*right)?;

                // Determine if float or int vector
                let is_float = match vec_ty {
                    IrType::Vector { element, .. } => {
                        matches!(element.as_ref(), IrType::F32 | IrType::F64)
                    }
                    _ => false,
                };

                let result = if is_float {
                    let lhs_vec = lhs.into_vector_value();
                    let rhs_vec = rhs.into_vector_value();
                    match op {
                        BinaryOp::Add | BinaryOp::FAdd => self
                            .builder
                            .build_float_add(lhs_vec, rhs_vec, "vadd")
                            .map_err(|e| format!("Vector fadd failed: {}", e))?
                            .into(),
                        BinaryOp::Sub | BinaryOp::FSub => self
                            .builder
                            .build_float_sub(lhs_vec, rhs_vec, "vsub")
                            .map_err(|e| format!("Vector fsub failed: {}", e))?
                            .into(),
                        BinaryOp::Mul | BinaryOp::FMul => self
                            .builder
                            .build_float_mul(lhs_vec, rhs_vec, "vmul")
                            .map_err(|e| format!("Vector fmul failed: {}", e))?
                            .into(),
                        BinaryOp::Div | BinaryOp::FDiv => self
                            .builder
                            .build_float_div(lhs_vec, rhs_vec, "vdiv")
                            .map_err(|e| format!("Vector fdiv failed: {}", e))?
                            .into(),
                        _ => return Err(format!("Unsupported float vector op: {:?}", op)),
                    }
                } else {
                    let lhs_vec = lhs.into_vector_value();
                    let rhs_vec = rhs.into_vector_value();
                    match op {
                        BinaryOp::Add => self
                            .builder
                            .build_int_add(lhs_vec, rhs_vec, "vadd")
                            .map_err(|e| format!("Vector iadd failed: {}", e))?
                            .into(),
                        BinaryOp::Sub => self
                            .builder
                            .build_int_sub(lhs_vec, rhs_vec, "vsub")
                            .map_err(|e| format!("Vector isub failed: {}", e))?
                            .into(),
                        BinaryOp::Mul => self
                            .builder
                            .build_int_mul(lhs_vec, rhs_vec, "vmul")
                            .map_err(|e| format!("Vector imul failed: {}", e))?
                            .into(),
                        BinaryOp::Div => self
                            .builder
                            .build_int_signed_div(lhs_vec, rhs_vec, "vdiv")
                            .map_err(|e| format!("Vector idiv failed: {}", e))?
                            .into(),
                        BinaryOp::And => self
                            .builder
                            .build_and(lhs_vec, rhs_vec, "vand")
                            .map_err(|e| format!("Vector and failed: {}", e))?
                            .into(),
                        BinaryOp::Or => self
                            .builder
                            .build_or(lhs_vec, rhs_vec, "vor")
                            .map_err(|e| format!("Vector or failed: {}", e))?
                            .into(),
                        BinaryOp::Xor => self
                            .builder
                            .build_xor(lhs_vec, rhs_vec, "vxor")
                            .map_err(|e| format!("Vector xor failed: {}", e))?
                            .into(),
                        _ => return Err(format!("Unsupported int vector op: {:?}", op)),
                    }
                };
                self.value_map.insert(*dest, result);
            }

            IrInstruction::VectorSplat {
                dest,
                scalar,
                vec_ty,
            } => {
                let scalar_val = self.get_value(*scalar)?;
                let vec_llvm_ty = self.translate_type(vec_ty)?;
                let vec_type = vec_llvm_ty.into_vector_type();
                let lane_count = vec_type.get_size();

                // Build splat by inserting scalar into all lanes
                let undef = vec_type.get_undef();
                let mut result: BasicValueEnum = undef.into();

                for i in 0..lane_count {
                    let idx = self.context.i32_type().const_int(i as u64, false);
                    result = self
                        .builder
                        .build_insert_element(
                            result.into_vector_value(),
                            scalar_val,
                            idx,
                            &format!("splat_{}", i),
                        )
                        .map_err(|e| format!("Vector splat insert failed: {}", e))?
                        .into();
                }
                self.value_map.insert(*dest, result);
            }

            IrInstruction::VectorExtract {
                dest,
                vector,
                index,
            } => {
                let vec_val = self.get_value(*vector)?.into_vector_value();
                let idx = self.context.i32_type().const_int(*index as u64, false);
                let extracted = self
                    .builder
                    .build_extract_element(vec_val, idx, &format!("extract_{}", dest.as_u32()))
                    .map_err(|e| format!("Vector extract failed: {}", e))?;
                self.value_map.insert(*dest, extracted);
            }

            IrInstruction::VectorInsert {
                dest,
                vector,
                scalar,
                index,
            } => {
                let vec_val = self.get_value(*vector)?.into_vector_value();
                let scalar_val = self.get_value(*scalar)?;
                let idx = self.context.i32_type().const_int(*index as u64, false);
                let result = self
                    .builder
                    .build_insert_element(
                        vec_val,
                        scalar_val,
                        idx,
                        &format!("insert_{}", dest.as_u32()),
                    )
                    .map_err(|e| format!("Vector insert failed: {}", e))?;
                self.value_map.insert(*dest, result.into());
            }

            IrInstruction::VectorReduce { dest, op, vector } => {
                let vec_val = self.get_value(*vector)?.into_vector_value();
                let vec_ty = vec_val.get_type();
                let lane_count = vec_ty.get_size();
                let elem_ty = vec_ty.get_element_type();
                let is_float = elem_ty.is_float_type();

                // Extract first element as accumulator
                let idx0 = self.context.i32_type().const_int(0, false);
                let mut acc = self
                    .builder
                    .build_extract_element(vec_val, idx0, "reduce_init")
                    .map_err(|e| format!("Reduce extract failed: {}", e))?;

                // Reduce remaining elements
                for i in 1..lane_count {
                    let idx = self.context.i32_type().const_int(i as u64, false);
                    let elem = self
                        .builder
                        .build_extract_element(vec_val, idx, &format!("reduce_{}", i))
                        .map_err(|e| format!("Reduce extract failed: {}", e))?;

                    acc = if is_float {
                        match op {
                            BinaryOp::Add | BinaryOp::FAdd => self
                                .builder
                                .build_float_add(
                                    acc.into_float_value(),
                                    elem.into_float_value(),
                                    "reduce_add",
                                )
                                .map_err(|e| format!("Reduce fadd failed: {}", e))?
                                .into(),
                            BinaryOp::Mul | BinaryOp::FMul => self
                                .builder
                                .build_float_mul(
                                    acc.into_float_value(),
                                    elem.into_float_value(),
                                    "reduce_mul",
                                )
                                .map_err(|e| format!("Reduce fmul failed: {}", e))?
                                .into(),
                            _ => return Err(format!("Unsupported float reduce op: {:?}", op)),
                        }
                    } else {
                        match op {
                            BinaryOp::Add => self
                                .builder
                                .build_int_add(
                                    acc.into_int_value(),
                                    elem.into_int_value(),
                                    "reduce_add",
                                )
                                .map_err(|e| format!("Reduce iadd failed: {}", e))?
                                .into(),
                            BinaryOp::Mul => self
                                .builder
                                .build_int_mul(
                                    acc.into_int_value(),
                                    elem.into_int_value(),
                                    "reduce_mul",
                                )
                                .map_err(|e| format!("Reduce imul failed: {}", e))?
                                .into(),
                            BinaryOp::And => self
                                .builder
                                .build_and(
                                    acc.into_int_value(),
                                    elem.into_int_value(),
                                    "reduce_and",
                                )
                                .map_err(|e| format!("Reduce and failed: {}", e))?
                                .into(),
                            BinaryOp::Or => self
                                .builder
                                .build_or(acc.into_int_value(), elem.into_int_value(), "reduce_or")
                                .map_err(|e| format!("Reduce or failed: {}", e))?
                                .into(),
                            BinaryOp::Xor => self
                                .builder
                                .build_xor(
                                    acc.into_int_value(),
                                    elem.into_int_value(),
                                    "reduce_xor",
                                )
                                .map_err(|e| format!("Reduce xor failed: {}", e))?
                                .into(),
                            _ => return Err(format!("Unsupported int reduce op: {:?}", op)),
                        }
                    };
                }
                self.value_map.insert(*dest, acc);
            }

            IrInstruction::VectorUnaryOp {
                dest,
                op,
                operand,
                vec_ty: _,
            } => {
                let operand_val = self.get_value(*operand)?.into_vector_value();
                let vec_ty = operand_val.get_type();
                let lane_count = vec_ty.get_size();
                let elem_ty = vec_ty.get_element_type().into_float_type();

                // Apply unary op lane-by-lane (LLVM intrinsics work on scalars)
                let mut result: inkwell::values::BasicValueEnum = operand_val.into();
                for i in 0..lane_count {
                    let idx = self.context.i32_type().const_int(i as u64, false);
                    let elem = self
                        .builder
                        .build_extract_element(operand_val, idx, &format!("unary_{}", i))
                        .map_err(|e| format!("VectorUnaryOp extract failed: {}", e))?
                        .into_float_value();

                    let processed = match op {
                        VectorUnaryOpKind::Sqrt => {
                            let intrinsic = inkwell::intrinsics::Intrinsic::find(&format!(
                                "llvm.sqrt.f{}",
                                if elem_ty == self.context.f32_type() {
                                    32
                                } else {
                                    64
                                }
                            ))
                            .ok_or("sqrt intrinsic not found")?;
                            let func = intrinsic
                                .get_declaration(&self.module, &[elem_ty.into()])
                                .ok_or("sqrt declaration failed")?;
                            self.builder
                                .build_call(func, &[elem.into()], "sqrt")
                                .map_err(|e| format!("sqrt call failed: {}", e))?
                                .try_as_basic_value()
                                .left()
                                .ok_or("sqrt returned void")?
                                .into_float_value()
                        }
                        VectorUnaryOpKind::Abs => {
                            let intrinsic = inkwell::intrinsics::Intrinsic::find(&format!(
                                "llvm.fabs.f{}",
                                if elem_ty == self.context.f32_type() {
                                    32
                                } else {
                                    64
                                }
                            ))
                            .ok_or("fabs intrinsic not found")?;
                            let func = intrinsic
                                .get_declaration(&self.module, &[elem_ty.into()])
                                .ok_or("fabs declaration failed")?;
                            self.builder
                                .build_call(func, &[elem.into()], "fabs")
                                .map_err(|e| format!("fabs call failed: {}", e))?
                                .try_as_basic_value()
                                .left()
                                .ok_or("fabs returned void")?
                                .into_float_value()
                        }
                        VectorUnaryOpKind::Neg => self
                            .builder
                            .build_float_neg(elem, "fneg")
                            .map_err(|e| format!("fneg failed: {}", e))?,
                        VectorUnaryOpKind::Ceil => {
                            let intrinsic = inkwell::intrinsics::Intrinsic::find(&format!(
                                "llvm.ceil.f{}",
                                if elem_ty == self.context.f32_type() {
                                    32
                                } else {
                                    64
                                }
                            ))
                            .ok_or("ceil intrinsic not found")?;
                            let func = intrinsic
                                .get_declaration(&self.module, &[elem_ty.into()])
                                .ok_or("ceil declaration failed")?;
                            self.builder
                                .build_call(func, &[elem.into()], "ceil")
                                .map_err(|e| format!("ceil call failed: {}", e))?
                                .try_as_basic_value()
                                .left()
                                .ok_or("ceil returned void")?
                                .into_float_value()
                        }
                        VectorUnaryOpKind::Floor => {
                            let intrinsic = inkwell::intrinsics::Intrinsic::find(&format!(
                                "llvm.floor.f{}",
                                if elem_ty == self.context.f32_type() {
                                    32
                                } else {
                                    64
                                }
                            ))
                            .ok_or("floor intrinsic not found")?;
                            let func = intrinsic
                                .get_declaration(&self.module, &[elem_ty.into()])
                                .ok_or("floor declaration failed")?;
                            self.builder
                                .build_call(func, &[elem.into()], "floor")
                                .map_err(|e| format!("floor call failed: {}", e))?
                                .try_as_basic_value()
                                .left()
                                .ok_or("floor returned void")?
                                .into_float_value()
                        }
                        VectorUnaryOpKind::Trunc => {
                            let intrinsic = inkwell::intrinsics::Intrinsic::find(&format!(
                                "llvm.trunc.f{}",
                                if elem_ty == self.context.f32_type() {
                                    32
                                } else {
                                    64
                                }
                            ))
                            .ok_or("trunc intrinsic not found")?;
                            let func = intrinsic
                                .get_declaration(&self.module, &[elem_ty.into()])
                                .ok_or("trunc declaration failed")?;
                            self.builder
                                .build_call(func, &[elem.into()], "trunc")
                                .map_err(|e| format!("trunc call failed: {}", e))?
                                .try_as_basic_value()
                                .left()
                                .ok_or("trunc returned void")?
                                .into_float_value()
                        }
                        VectorUnaryOpKind::Round => {
                            let intrinsic = inkwell::intrinsics::Intrinsic::find(&format!(
                                "llvm.round.f{}",
                                if elem_ty == self.context.f32_type() {
                                    32
                                } else {
                                    64
                                }
                            ))
                            .ok_or("round intrinsic not found")?;
                            let func = intrinsic
                                .get_declaration(&self.module, &[elem_ty.into()])
                                .ok_or("round declaration failed")?;
                            self.builder
                                .build_call(func, &[elem.into()], "round")
                                .map_err(|e| format!("round call failed: {}", e))?
                                .try_as_basic_value()
                                .left()
                                .ok_or("round returned void")?
                                .into_float_value()
                        }
                    };

                    result = self
                        .builder
                        .build_insert_element(
                            result.into_vector_value(),
                            processed,
                            idx,
                            &format!("unary_insert_{}", i),
                        )
                        .map_err(|e| format!("VectorUnaryOp insert failed: {}", e))?
                        .into();
                }
                self.value_map.insert(*dest, result);
            }

            IrInstruction::VectorMinMax {
                dest,
                op,
                left,
                right,
                vec_ty: _,
            } => {
                let lhs = self.get_value(*left)?.into_vector_value();
                let rhs = self.get_value(*right)?.into_vector_value();
                let vec_ty = lhs.get_type();
                let lane_count = vec_ty.get_size();
                let elem_ty = vec_ty.get_element_type().into_float_type();

                let intrinsic_name = match op {
                    VectorMinMaxKind::Min => format!(
                        "llvm.minnum.f{}",
                        if elem_ty == self.context.f32_type() {
                            32
                        } else {
                            64
                        }
                    ),
                    VectorMinMaxKind::Max => format!(
                        "llvm.maxnum.f{}",
                        if elem_ty == self.context.f32_type() {
                            32
                        } else {
                            64
                        }
                    ),
                };

                let intrinsic = inkwell::intrinsics::Intrinsic::find(&intrinsic_name)
                    .ok_or(format!("{} not found", intrinsic_name))?;
                let func = intrinsic
                    .get_declaration(&self.module, &[elem_ty.into()])
                    .ok_or(format!("{} declaration failed", intrinsic_name))?;

                let mut result: inkwell::values::BasicValueEnum = lhs.into();
                for i in 0..lane_count {
                    let idx = self.context.i32_type().const_int(i as u64, false);
                    let l = self
                        .builder
                        .build_extract_element(lhs, idx, &format!("minmax_l_{}", i))
                        .map_err(|e| format!("MinMax extract failed: {}", e))?;
                    let r = self
                        .builder
                        .build_extract_element(rhs, idx, &format!("minmax_r_{}", i))
                        .map_err(|e| format!("MinMax extract failed: {}", e))?;
                    let val = self
                        .builder
                        .build_call(func, &[l.into(), r.into()], "minmax")
                        .map_err(|e| format!("MinMax call failed: {}", e))?
                        .try_as_basic_value()
                        .left()
                        .ok_or("minmax returned void")?;
                    result = self
                        .builder
                        .build_insert_element(
                            result.into_vector_value(),
                            val.into_float_value(),
                            idx,
                            &format!("minmax_insert_{}", i),
                        )
                        .map_err(|e| format!("MinMax insert failed: {}", e))?
                        .into();
                }
                self.value_map.insert(*dest, result);
            }

            // Global variable access - inline load from LLVM global (no FFI)
            IrInstruction::LoadGlobal {
                dest,
                global_id,
                ty,
            } => {
                // Get or create LLVM global variable for this global_id
                let global = self.get_or_create_global(*global_id);
                let global_ptr = global.as_pointer_value();

                // Load the i64 value directly from the global
                let result = self
                    .builder
                    .build_load(
                        self.context.i64_type(),
                        global_ptr,
                        &format!("global_{}", global_id.0),
                    )
                    .map_err(|e| format!("Failed to load global: {}", e))?;

                // Cast the i64 result to the expected type if needed
                let llvm_ty = self.translate_type(ty)?;
                let final_val = if llvm_ty.is_pointer_type() {
                    self.builder
                        .build_int_to_ptr(
                            result.into_int_value(),
                            llvm_ty.into_pointer_type(),
                            &format!("global_ptr_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to cast global to ptr: {}", e))?
                        .into()
                } else if llvm_ty.is_float_type() {
                    self.builder
                        .build_bit_cast(
                            result.into_int_value(),
                            llvm_ty,
                            &format!("global_float_{}", dest.as_u32()),
                        )
                        .map_err(|e| format!("Failed to cast global to float: {}", e))?
                } else {
                    result
                };
                self.value_map.insert(*dest, final_val);
            }

            // Global variable store - inline store to LLVM global (no FFI)
            IrInstruction::StoreGlobal { global_id, value } => {
                // Get or create LLVM global variable for this global_id
                let global = self.get_or_create_global(*global_id);
                let global_ptr = global.as_pointer_value();

                let raw_val = self.get_value(*value)?;

                // Convert value to i64 for storage
                let val_i64: inkwell::values::IntValue = if raw_val.is_pointer_value() {
                    self.builder
                        .build_ptr_to_int(
                            raw_val.into_pointer_value(),
                            self.context.i64_type(),
                            "global_store_ptrtoint",
                        )
                        .map_err(|e| format!("Failed to cast ptr for global store: {}", e))?
                } else if raw_val.is_float_value() {
                    self.builder
                        .build_bit_cast(
                            raw_val.into_float_value(),
                            self.context.i64_type(),
                            "global_store_float",
                        )
                        .map_err(|e| format!("Failed to cast float for global store: {}", e))?
                        .into_int_value()
                } else if raw_val.is_int_value() {
                    // May need to extend smaller ints to i64
                    let int_val = raw_val.into_int_value();
                    if int_val.get_type().get_bit_width() < 64 {
                        self.builder
                            .build_int_z_extend(
                                int_val,
                                self.context.i64_type(),
                                "global_store_zext",
                            )
                            .map_err(|e| format!("Failed to extend int for global store: {}", e))?
                    } else {
                        int_val
                    }
                } else {
                    return Err(format!("Cannot store {:?} to global", raw_val));
                };

                // Store directly to the LLVM global
                self.builder
                    .build_store(global_ptr, val_i64)
                    .map_err(|e| format!("Failed to store to global: {}", e))?;
            }

            // Panic
            IrInstruction::Panic { .. } => {
                // Build a trap/abort
                self.builder
                    .build_unreachable()
                    .map_err(|e| format!("Failed to build panic: {}", e))?;
            }
        }

        Ok(())
    }

    /// Compile a terminator instruction
    fn compile_terminator(
        &mut self,
        term: &IrTerminator,
        llvm_func: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        match term {
            IrTerminator::Return { value } => {
                // Check if this function uses sret (struct return via pointer)
                if let Some(sret_ptr) = self.current_sret_ptr {
                    // sret function: store return value through the sret pointer, then return void
                    if let Some(val_id) = value {
                        let return_val = self.get_value(*val_id)?;
                        self.builder
                            .build_store(sret_ptr, return_val)
                            .map_err(|e| format!("Failed to store sret return value: {}", e))?;
                    }
                    // Always return void for sret functions
                    self.builder
                        .build_return(None)
                        .map_err(|e| format!("Failed to build sret void return: {}", e))?;
                } else if let Some(val_id) = value {
                    // Normal return (non-sret)
                    let return_val = self.get_value(*val_id)?;

                    // Get expected return type from the function
                    let expected_ret_ty = llvm_func.get_type().get_return_type();

                    // Coerce return value to match expected type if needed
                    let coerced_val = if let Some(expected) = expected_ret_ty {
                        let actual_ty = return_val.get_type();
                        if actual_ty == expected {
                            return_val
                        } else {
                            // Type mismatch - try to coerce
                            let cast_name = format!("ret_cast_{}", val_id.as_u32());

                            // Handle ptr -> struct conversion (e.g., ptr -> {i64, ptr})
                            if return_val.is_pointer_value() && expected.is_struct_type() {
                                // Wrap pointer in expected struct type
                                // For Array type {i64, ptr}: create struct with 0 length and the pointer
                                let struct_ty = expected.into_struct_type();
                                let len_val = self.context.i64_type().const_int(0, false);
                                let ptr_val = return_val.into_pointer_value();
                                let s1 = self
                                    .builder
                                    .build_insert_value(
                                        struct_ty.const_zero(),
                                        len_val,
                                        0,
                                        &format!("{}_len", cast_name),
                                    )
                                    .map_err(|e| {
                                        format!("Failed to build struct for return: {}", e)
                                    })?
                                    .into_struct_value();
                                let s2 = self
                                    .builder
                                    .build_insert_value(
                                        s1,
                                        ptr_val,
                                        1,
                                        &format!("{}_ptr", cast_name),
                                    )
                                    .map_err(|e| {
                                        format!("Failed to build struct for return: {}", e)
                                    })?
                                    .into_struct_value();
                                s2.into()
                            } else if return_val.is_float_value() && expected.is_int_type() {
                                // Float to int
                                self.builder
                                    .build_float_to_signed_int(
                                        return_val.into_float_value(),
                                        expected.into_int_type(),
                                        &cast_name,
                                    )
                                    .map_err(|e| format!("Failed to cast return: {}", e))?
                                    .into()
                            } else if return_val.is_int_value() && expected.is_float_type() {
                                // Int to float
                                self.builder
                                    .build_signed_int_to_float(
                                        return_val.into_int_value(),
                                        expected.into_float_type(),
                                        &cast_name,
                                    )
                                    .map_err(|e| format!("Failed to cast return: {}", e))?
                                    .into()
                            } else {
                                // Use as-is, let LLVM report the error
                                return_val
                            }
                        }
                    } else {
                        return_val
                    };

                    self.builder
                        .build_return(Some(&coerced_val))
                        .map_err(|e| format!("Failed to build return: {}", e))?;
                } else {
                    self.builder
                        .build_return(None)
                        .map_err(|e| format!("Failed to build void return: {}", e))?;
                }
            }

            IrTerminator::Branch { target } => {
                let target_block = self
                    .block_map
                    .get(target)
                    .ok_or_else(|| format!("Target block {:?} not found", target))?;
                self.builder
                    .build_unconditional_branch(*target_block)
                    .map_err(|e| format!("Failed to build branch: {}", e))?;
            }

            IrTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let cond_raw = self.get_value(*condition)?;
                // LLVM requires branch conditions to be i1, but our Bool type is i8
                // Convert to i1 by comparing with 0 (any non-zero value is true)
                let cond_val = if cond_raw.is_float_value() {
                    // Convert float to bool (non-zero = true)
                    let zero = self.context.f64_type().const_float(0.0);
                    self.builder
                        .build_float_compare(
                            inkwell::FloatPredicate::ONE,
                            cond_raw.into_float_value(),
                            zero,
                            "cond_bool",
                        )
                        .map_err(|e| format!("Failed to convert cond: {}", e))?
                } else {
                    let int_val = cond_raw.into_int_value();
                    // If it's already i1, use it directly; otherwise compare with 0
                    if int_val.get_type().get_bit_width() == 1 {
                        int_val
                    } else {
                        // Compare with 0 to get i1 (ne 0 = true)
                        let zero = int_val.get_type().const_int(0, false);
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::NE, int_val, zero, "cond_i1")
                            .map_err(|e| format!("Failed to convert cond to i1: {}", e))?
                    }
                };
                let true_block = self
                    .block_map
                    .get(true_target)
                    .ok_or_else(|| format!("True target block {:?} not found", true_target))?;
                let false_block = self
                    .block_map
                    .get(false_target)
                    .ok_or_else(|| format!("False target block {:?} not found", false_target))?;

                self.builder
                    .build_conditional_branch(cond_val, *true_block, *false_block)
                    .map_err(|e| format!("Failed to build conditional branch: {}", e))?;
            }

            IrTerminator::Switch {
                value,
                cases,
                default,
            } => {
                let switch_raw = self.get_value(*value)?;
                let switch_val = if switch_raw.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            switch_raw.into_float_value(),
                            self.context.i64_type(),
                            "switch_cast",
                        )
                        .map_err(|e| format!("Failed to cast switch value: {}", e))?
                } else {
                    switch_raw.into_int_value()
                };
                let default_block = self
                    .block_map
                    .get(default)
                    .ok_or_else(|| format!("Default block {:?} not found", default))?;

                // Build the cases vector for LLVM
                let llvm_cases: Result<Vec<_>, String> = cases
                    .iter()
                    .map(|(case_val, case_target)| -> Result<_, String> {
                        let case_block = self.block_map.get(case_target).ok_or_else(|| {
                            format!("Case target block {:?} not found", case_target)
                        })?;
                        let const_val = self.context.i64_type().const_int(*case_val as u64, false);
                        Ok((const_val, *case_block))
                    })
                    .collect();
                let llvm_cases = llvm_cases?;

                self.builder
                    .build_switch(switch_val, *default_block, &llvm_cases)
                    .map_err(|e| format!("Failed to build switch: {}", e))?;
            }

            IrTerminator::Unreachable => {
                self.builder
                    .build_unreachable()
                    .map_err(|e| format!("Failed to build unreachable: {}", e))?;
            }

            IrTerminator::NoReturn { .. } => {
                self.builder
                    .build_unreachable()
                    .map_err(|e| format!("Failed to build unreachable (no return): {}", e))?;
            }
        }

        Ok(())
    }

    /// Get an LLVM value from the value map
    fn get_value(&self, id: IrId) -> Result<BasicValueEnum<'ctx>, String> {
        if let Some(val) = self.value_map.get(&id) {
            return Ok(*val);
        }

        // Value not found - this might be a forward reference or missing instruction
        // For now, return a detailed error
        let mut available: Vec<_> = self.value_map.keys().map(|k| k.as_u32()).collect();
        available.sort();
        Err(format!(
            "Value {:?} not found in value map. Available IrIds: {:?}",
            id, available
        ))
    }

    /// Compile a constant value
    fn compile_constant(&self, value: &IrValue) -> Result<BasicValueEnum<'ctx>, String> {
        match value {
            IrValue::Void | IrValue::Undef => Err("Cannot compile void/undef as value".to_string()),
            IrValue::Null => Ok(self
                .context
                .ptr_type(AddressSpace::default())
                .const_null()
                .into()),
            IrValue::Bool(b) => {
                // Our Bool type is i8, not i1
                Ok(self.context.i8_type().const_int(*b as u64, false).into())
            }
            IrValue::I8(v) => Ok(self.context.i8_type().const_int(*v as u64, true).into()),
            IrValue::I16(v) => Ok(self.context.i16_type().const_int(*v as u64, true).into()),
            IrValue::I32(v) => Ok(self.context.i32_type().const_int(*v as u64, true).into()),
            IrValue::I64(v) => Ok(self.context.i64_type().const_int(*v as u64, true).into()),
            IrValue::U8(v) => Ok(self.context.i8_type().const_int(*v as u64, false).into()),
            IrValue::U16(v) => Ok(self.context.i16_type().const_int(*v as u64, false).into()),
            IrValue::U32(v) => Ok(self.context.i32_type().const_int(*v as u64, false).into()),
            IrValue::U64(v) => Ok(self.context.i64_type().const_int(*v, false).into()),
            IrValue::F32(v) => Ok(self.context.f32_type().const_float(*v as f64).into()),
            IrValue::F64(v) => Ok(self.context.f64_type().const_float(*v).into()),
            IrValue::String(s) => {
                // Create global string constant with the raw bytes
                let global_str = self
                    .builder
                    .build_global_string_ptr(s, "str")
                    .map_err(|e| format!("Failed to build global string: {}", e))?;
                let str_ptr = global_str.as_pointer_value();
                let str_len = self.context.i64_type().const_int(s.len() as u64, false);

                // Get or declare haxe_string_literal(ptr, len) -> *mut HaxeString
                let string_literal_fn = match self.module.get_function("haxe_string_literal") {
                    Some(f) => f,
                    None => {
                        let fn_type = self.context.ptr_type(AddressSpace::default()).fn_type(
                            &[
                                self.context.ptr_type(AddressSpace::default()).into(),
                                self.context.i64_type().into(),
                            ],
                            false,
                        );
                        self.module
                            .add_function("haxe_string_literal", fn_type, None)
                    }
                };

                // Call haxe_string_literal(ptr, len) -> *mut HaxeString
                let result = self
                    .builder
                    .build_call(
                        string_literal_fn,
                        &[str_ptr.into(), str_len.into()],
                        "str_literal",
                    )
                    .map_err(|e| format!("Failed to call haxe_string_literal: {}", e))?;

                let haxe_str_ptr = result
                    .try_as_basic_value()
                    .left()
                    .ok_or("haxe_string_literal did not return a value")?;

                Ok(haxe_str_ptr)
            }
            IrValue::Array(_)
            | IrValue::Struct(_)
            | IrValue::Function(_)
            | IrValue::Closure { .. } => {
                Err("Complex constant values not yet supported".to_string())
            }
        }
    }

    /// Compile binary operation
    /// The result_ty is the MIR type for the result, used to determine integer vs float ops
    fn compile_binop(
        &self,
        op: BinaryOp,
        left: BasicValueEnum<'ctx>,
        right: BasicValueEnum<'ctx>,
        dest: IrId,
        result_ty: Option<&IrType>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let name = format!("binop_{}", dest.as_u32());

        // Determine if this is a float operation
        // Primary: use MIR type if available
        // Safety net: also check LLVM values in case MIR type is wrong (e.g., defaulted to I64)
        let is_float = match result_ty {
            Some(ty) if ty.is_float() => true,
            Some(_) => {
                // MIR says non-float, but check actual LLVM values as safety net
                // This catches cases where register_types defaulted to I64
                left.is_float_value() || right.is_float_value()
            }
            // Fallback to LLVM value inference if MIR type not available
            None => left.is_float_value() || right.is_float_value(),
        };

        // Helper function to convert int to float if needed
        let to_float = |val: BasicValueEnum<'ctx>,
                        builder: &inkwell::builder::Builder<'ctx>,
                        name: &str|
         -> Result<inkwell::values::FloatValue<'ctx>, String> {
            if val.is_float_value() {
                Ok(val.into_float_value())
            } else {
                builder
                    .build_signed_int_to_float(val.into_int_value(), self.context.f64_type(), name)
                    .map_err(|e| format!("Failed to convert int to float: {}", e))
            }
        };

        match op {
            // Arithmetic - dispatch based on operand type
            BinaryOp::Add => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "add_l_f")?;
                    let right_f = to_float(right, &self.builder, "add_r_f")?;
                    // Fuse multiply-add: fadd(fmul(a, b), c) → fma(a, b, c)
                    let result = if let Some((a, b)) = self.try_extract_fmul_llvm(left_f) {
                        self.build_fma(a, b, right_f, &name)?
                    } else if let Some((a, b)) = self.try_extract_fmul_llvm(right_f) {
                        self.build_fma(a, b, left_f, &name)?
                    } else {
                        let r = self
                            .builder
                            .build_float_add(left_f, right_f, &name)
                            .map_err(|e| format!("Failed to build fadd: {}", e))?;
                        self.apply_fast_math(r);
                        r
                    };
                    Ok(result.into())
                } else {
                    let result = self
                        .builder
                        .build_int_add(left.into_int_value(), right.into_int_value(), &name)
                        .map_err(|e| format!("Failed to build add: {}", e))?;
                    Ok(result.into())
                }
            }
            BinaryOp::Sub => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "sub_l_f")?;
                    let right_f = to_float(right, &self.builder, "sub_r_f")?;
                    // Fuse multiply-subtract: fsub(fmul(a, b), c) → fma(a, b, fneg(c))
                    let result = if let Some((a, b)) = self.try_extract_fmul_llvm(left_f) {
                        let neg_rhs = self
                            .builder
                            .build_float_neg(right_f, "neg_rhs")
                            .map_err(|e| format!("Failed to build fneg: {}", e))?;
                        self.build_fma(a, b, neg_rhs, &name)?
                    } else if let Some((a, b)) = self.try_extract_fmul_llvm(right_f) {
                        let neg_a = self
                            .builder
                            .build_float_neg(a, "neg_a")
                            .map_err(|e| format!("Failed to build fneg: {}", e))?;
                        self.build_fma(neg_a, b, left_f, &name)?
                    } else {
                        let r = self
                            .builder
                            .build_float_sub(left_f, right_f, &name)
                            .map_err(|e| format!("Failed to build fsub: {}", e))?;
                        self.apply_fast_math(r);
                        r
                    };
                    Ok(result.into())
                } else {
                    let result = self
                        .builder
                        .build_int_sub(left.into_int_value(), right.into_int_value(), &name)
                        .map_err(|e| format!("Failed to build sub: {}", e))?;
                    Ok(result.into())
                }
            }
            BinaryOp::Mul => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "mul_l_f")?;
                    let right_f = to_float(right, &self.builder, "mul_r_f")?;
                    let result = self
                        .builder
                        .build_float_mul(left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build fmul: {}", e))?;
                    self.apply_fast_math(result);
                    Ok(result.into())
                } else {
                    let result = self
                        .builder
                        .build_int_mul(left.into_int_value(), right.into_int_value(), &name)
                        .map_err(|e| format!("Failed to build mul: {}", e))?;
                    Ok(result.into())
                }
            }
            BinaryOp::Div => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "div_l_f")?;
                    let right_f = to_float(right, &self.builder, "div_r_f")?;
                    let result = self
                        .builder
                        .build_float_div(left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build fdiv: {}", e))?;
                    self.apply_fast_math(result);
                    Ok(result.into())
                } else {
                    let result = self
                        .builder
                        .build_int_signed_div(left.into_int_value(), right.into_int_value(), &name)
                        .map_err(|e| format!("Failed to build div: {}", e))?;
                    Ok(result.into())
                }
            }
            BinaryOp::Rem => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "rem_l_f")?;
                    let right_f = to_float(right, &self.builder, "rem_r_f")?;
                    let result = self
                        .builder
                        .build_float_rem(left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build frem: {}", e))?;
                    self.apply_fast_math(result);
                    Ok(result.into())
                } else {
                    let result = self
                        .builder
                        .build_int_signed_rem(left.into_int_value(), right.into_int_value(), &name)
                        .map_err(|e| format!("Failed to build rem: {}", e))?;
                    Ok(result.into())
                }
            }

            // Bitwise operations - convert floats to int if needed
            BinaryOp::And => {
                let left_int = if left.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            left.into_float_value(),
                            self.context.i64_type(),
                            "and_l_cast",
                        )
                        .map_err(|e| format!("Failed to cast and left: {}", e))?
                } else {
                    left.into_int_value()
                };
                let right_int = if right.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            right.into_float_value(),
                            self.context.i64_type(),
                            "and_r_cast",
                        )
                        .map_err(|e| format!("Failed to cast and right: {}", e))?
                } else {
                    right.into_int_value()
                };
                let result = self
                    .builder
                    .build_and(left_int, right_int, &name)
                    .map_err(|e| format!("Failed to build and: {}", e))?;
                Ok(result.into())
            }
            BinaryOp::Or => {
                let left_int = if left.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            left.into_float_value(),
                            self.context.i64_type(),
                            "or_l_cast",
                        )
                        .map_err(|e| format!("Failed to cast or left: {}", e))?
                } else {
                    left.into_int_value()
                };
                let right_int = if right.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            right.into_float_value(),
                            self.context.i64_type(),
                            "or_r_cast",
                        )
                        .map_err(|e| format!("Failed to cast or right: {}", e))?
                } else {
                    right.into_int_value()
                };
                let result = self
                    .builder
                    .build_or(left_int, right_int, &name)
                    .map_err(|e| format!("Failed to build or: {}", e))?;
                Ok(result.into())
            }
            BinaryOp::Xor => {
                let left_int = if left.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            left.into_float_value(),
                            self.context.i64_type(),
                            "xor_l_cast",
                        )
                        .map_err(|e| format!("Failed to cast xor left: {}", e))?
                } else {
                    left.into_int_value()
                };
                let right_int = if right.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            right.into_float_value(),
                            self.context.i64_type(),
                            "xor_r_cast",
                        )
                        .map_err(|e| format!("Failed to cast xor right: {}", e))?
                } else {
                    right.into_int_value()
                };
                let result = self
                    .builder
                    .build_xor(left_int, right_int, &name)
                    .map_err(|e| format!("Failed to build xor: {}", e))?;
                Ok(result.into())
            }
            BinaryOp::Shl => {
                let left_int = if left.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            left.into_float_value(),
                            self.context.i64_type(),
                            "shl_l_cast",
                        )
                        .map_err(|e| format!("Failed to cast shl left: {}", e))?
                } else {
                    left.into_int_value()
                };
                let right_int = if right.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            right.into_float_value(),
                            self.context.i64_type(),
                            "shl_r_cast",
                        )
                        .map_err(|e| format!("Failed to cast shl right: {}", e))?
                } else {
                    right.into_int_value()
                };
                let result = self
                    .builder
                    .build_left_shift(left_int, right_int, &name)
                    .map_err(|e| format!("Failed to build shl: {}", e))?;
                Ok(result.into())
            }
            BinaryOp::Shr => {
                let left_int = if left.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            left.into_float_value(),
                            self.context.i64_type(),
                            "shr_l_cast",
                        )
                        .map_err(|e| format!("Failed to cast shr left: {}", e))?
                } else {
                    left.into_int_value()
                };
                let right_int = if right.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            right.into_float_value(),
                            self.context.i64_type(),
                            "shr_r_cast",
                        )
                        .map_err(|e| format!("Failed to cast shr right: {}", e))?
                } else {
                    right.into_int_value()
                };
                let result = self
                    .builder
                    .build_right_shift(left_int, right_int, true, &name)
                    .map_err(|e| format!("Failed to build shr: {}", e))?;
                Ok(result.into())
            }

            // Float arithmetic (explicit float operations)
            BinaryOp::FAdd => {
                let left_f = left.into_float_value();
                let right_f = right.into_float_value();
                // Fuse multiply-add: fadd(fmul(a, b), c) → fma(a, b, c)
                let result = if let Some((a, b)) = self.try_extract_fmul_llvm(left_f) {
                    self.build_fma(a, b, right_f, &name)?
                } else if let Some((a, b)) = self.try_extract_fmul_llvm(right_f) {
                    self.build_fma(a, b, left_f, &name)?
                } else {
                    let r = self
                        .builder
                        .build_float_add(left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build fadd: {}", e))?;
                    self.apply_fast_math(r);
                    r
                };
                Ok(result.into())
            }
            BinaryOp::FSub => {
                let left_f = left.into_float_value();
                let right_f = right.into_float_value();
                // Fuse multiply-subtract: fsub(fmul(a, b), c) → fma(a, b, fneg(c))
                let result = if let Some((a, b)) = self.try_extract_fmul_llvm(left_f) {
                    let neg_rhs = self
                        .builder
                        .build_float_neg(right_f, "neg_rhs")
                        .map_err(|e| format!("Failed to build fneg: {}", e))?;
                    self.build_fma(a, b, neg_rhs, &name)?
                } else if let Some((a, b)) = self.try_extract_fmul_llvm(right_f) {
                    let neg_a = self
                        .builder
                        .build_float_neg(a, "neg_a")
                        .map_err(|e| format!("Failed to build fneg: {}", e))?;
                    self.build_fma(neg_a, b, left_f, &name)?
                } else {
                    let r = self
                        .builder
                        .build_float_sub(left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build fsub: {}", e))?;
                    self.apply_fast_math(r);
                    r
                };
                Ok(result.into())
            }
            BinaryOp::FMul => {
                let result = self
                    .builder
                    .build_float_mul(left.into_float_value(), right.into_float_value(), &name)
                    .map_err(|e| format!("Failed to build fmul: {}", e))?;
                self.apply_fast_math(result);
                Ok(result.into())
            }
            BinaryOp::FDiv => {
                let result = self
                    .builder
                    .build_float_div(left.into_float_value(), right.into_float_value(), &name)
                    .map_err(|e| format!("Failed to build fdiv: {}", e))?;
                self.apply_fast_math(result);
                Ok(result.into())
            }
            BinaryOp::FRem => {
                let result = self
                    .builder
                    .build_float_rem(left.into_float_value(), right.into_float_value(), &name)
                    .map_err(|e| format!("Failed to build frem: {}", e))?;
                self.apply_fast_math(result);
                Ok(result.into())
            }
        }
    }

    /// Compile unary operation
    fn compile_unop(
        &self,
        op: UnaryOp,
        operand: BasicValueEnum<'ctx>,
        dest: IrId,
        result_ty: Option<&IrType>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let name = format!("unop_{}", dest.as_u32());

        // Determine if this is a float operation
        // Primary: use MIR type if available
        // Safety net: also check LLVM value in case MIR type is wrong
        let is_float = match result_ty {
            Some(ty) if ty.is_float() => true,
            Some(_) => operand.is_float_value(),
            None => operand.is_float_value(),
        };

        match op {
            UnaryOp::Neg => {
                // Neg can be applied to both int and float
                if is_float {
                    let result = self
                        .builder
                        .build_float_neg(operand.into_float_value(), &name)
                        .map_err(|e| format!("Failed to build fneg: {}", e))?;
                    self.apply_fast_math(result);
                    Ok(result.into())
                } else {
                    let result = self
                        .builder
                        .build_int_neg(operand.into_int_value(), &name)
                        .map_err(|e| format!("Failed to build neg: {}", e))?;
                    Ok(result.into())
                }
            }
            UnaryOp::Not => {
                // Not is only for integers/booleans - convert float if needed
                let int_val = if operand.is_float_value() {
                    self.builder
                        .build_float_to_signed_int(
                            operand.into_float_value(),
                            self.context.i64_type(),
                            "not_cast",
                        )
                        .map_err(|e| format!("Failed to cast not operand: {}", e))?
                } else {
                    operand.into_int_value()
                };
                let result = self
                    .builder
                    .build_not(int_val, &name)
                    .map_err(|e| format!("Failed to build not: {}", e))?;
                Ok(result.into())
            }
            UnaryOp::FNeg => {
                let result = self
                    .builder
                    .build_float_neg(operand.into_float_value(), &name)
                    .map_err(|e| format!("Failed to build fneg: {}", e))?;
                self.apply_fast_math(result);
                Ok(result.into())
            }
        }
    }

    /// Compile comparison operation
    /// Note: Comparisons return i8 (our Bool type), not i1 (LLVM's native bool)
    fn compile_compare(
        &self,
        op: CompareOp,
        left: BasicValueEnum<'ctx>,
        right: BasicValueEnum<'ctx>,
        dest: IrId,
        operand_ty: Option<&IrType>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let name = format!("cmp_{}", dest.as_u32());

        // Determine if operands are float
        // Primary: use MIR type if available
        // Safety net: also check LLVM values in case MIR type is wrong
        let is_float = match operand_ty {
            Some(ty) if ty.is_float() => true,
            Some(_) => left.is_float_value() || right.is_float_value(),
            None => left.is_float_value() || right.is_float_value(),
        };

        // Helper to convert int to float if needed
        let to_float = |val: BasicValueEnum<'ctx>,
                        builder: &inkwell::builder::Builder<'ctx>,
                        n: &str|
         -> Result<inkwell::values::FloatValue<'ctx>, String> {
            if val.is_float_value() {
                Ok(val.into_float_value())
            } else {
                builder
                    .build_signed_int_to_float(val.into_int_value(), self.context.f64_type(), n)
                    .map_err(|e| format!("Failed to convert int to float: {}", e))
            }
        };

        // Helper to extend i1 comparison result to i8 (our Bool type)
        let to_i8 = |result: inkwell::values::IntValue<'ctx>,
                     builder: &inkwell::builder::Builder<'ctx>,
                     n: &str|
         -> Result<BasicValueEnum<'ctx>, String> {
            let ext = builder
                .build_int_z_extend(result, self.context.i8_type(), n)
                .map_err(|e| format!("Failed to extend bool to i8: {}", e))?;
            Ok(ext.into())
        };

        // Helper to coerce two integer operands to the same bit width.
        // LLVM's icmp requires both operands to have the same type.
        // If widths differ (e.g., i32 vs i64), sign-extend the narrower one.
        let coerce_ints = |l: BasicValueEnum<'ctx>,
                           r: BasicValueEnum<'ctx>,
                           builder: &inkwell::builder::Builder<'ctx>|
         -> (
            inkwell::values::IntValue<'ctx>,
            inkwell::values::IntValue<'ctx>,
        ) {
            let li = l.into_int_value();
            let ri = r.into_int_value();
            let lw = li.get_type().get_bit_width();
            let rw = ri.get_type().get_bit_width();
            if lw == rw {
                (li, ri)
            } else if lw < rw {
                let ext = builder
                    .build_int_s_extend(li, ri.get_type(), "cmp_sext_l")
                    .unwrap();
                (ext, ri)
            } else {
                let ext = builder
                    .build_int_s_extend(ri, li.get_type(), "cmp_sext_r")
                    .unwrap();
                (li, ext)
            }
        };

        match op {
            // Integer/Float comparisons - dispatch based on operand type
            CompareOp::Eq => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "eq_l_f")?;
                    let right_f = to_float(right, &self.builder, "eq_r_f")?;
                    let result = self
                        .builder
                        .build_float_compare(FloatPredicate::OEQ, left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build feq: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                } else {
                    let (li, ri) = coerce_ints(left, right, &self.builder);
                    let result = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, li, ri, &name)
                        .map_err(|e| format!("Failed to build eq: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                }
            }
            CompareOp::Ne => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "ne_l_f")?;
                    let right_f = to_float(right, &self.builder, "ne_r_f")?;
                    let result = self
                        .builder
                        .build_float_compare(FloatPredicate::ONE, left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build fne: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                } else {
                    let (li, ri) = coerce_ints(left, right, &self.builder);
                    let result = self
                        .builder
                        .build_int_compare(IntPredicate::NE, li, ri, &name)
                        .map_err(|e| format!("Failed to build ne: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                }
            }
            CompareOp::Lt => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "lt_l_f")?;
                    let right_f = to_float(right, &self.builder, "lt_r_f")?;
                    let result = self
                        .builder
                        .build_float_compare(FloatPredicate::OLT, left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build flt: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                } else {
                    let (li, ri) = coerce_ints(left, right, &self.builder);
                    let result = self
                        .builder
                        .build_int_compare(IntPredicate::SLT, li, ri, &name)
                        .map_err(|e| format!("Failed to build lt: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                }
            }
            CompareOp::Le => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "le_l_f")?;
                    let right_f = to_float(right, &self.builder, "le_r_f")?;
                    let result = self
                        .builder
                        .build_float_compare(FloatPredicate::OLE, left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build fle: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                } else {
                    let (li, ri) = coerce_ints(left, right, &self.builder);
                    let result = self
                        .builder
                        .build_int_compare(IntPredicate::SLE, li, ri, &name)
                        .map_err(|e| format!("Failed to build le: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                }
            }
            CompareOp::Gt => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "gt_l_f")?;
                    let right_f = to_float(right, &self.builder, "gt_r_f")?;
                    let result = self
                        .builder
                        .build_float_compare(FloatPredicate::OGT, left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build fgt: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                } else {
                    let (li, ri) = coerce_ints(left, right, &self.builder);
                    let result = self
                        .builder
                        .build_int_compare(IntPredicate::SGT, li, ri, &name)
                        .map_err(|e| format!("Failed to build gt: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                }
            }
            CompareOp::Ge => {
                if is_float {
                    let left_f = to_float(left, &self.builder, "ge_l_f")?;
                    let right_f = to_float(right, &self.builder, "ge_r_f")?;
                    let result = self
                        .builder
                        .build_float_compare(FloatPredicate::OGE, left_f, right_f, &name)
                        .map_err(|e| format!("Failed to build fge: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                } else {
                    let (li, ri) = coerce_ints(left, right, &self.builder);
                    let result = self
                        .builder
                        .build_int_compare(IntPredicate::SGE, li, ri, &name)
                        .map_err(|e| format!("Failed to build ge: {}", e))?;
                    to_i8(result, &self.builder, &format!("{}_i8", name))
                }
            }

            // Unsigned comparisons
            CompareOp::ULt => {
                let (li, ri) = coerce_ints(left, right, &self.builder);
                let result = self
                    .builder
                    .build_int_compare(IntPredicate::ULT, li, ri, &name)
                    .map_err(|e| format!("Failed to build ult: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::ULe => {
                let (li, ri) = coerce_ints(left, right, &self.builder);
                let result = self
                    .builder
                    .build_int_compare(IntPredicate::ULE, li, ri, &name)
                    .map_err(|e| format!("Failed to build ule: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::UGt => {
                let (li, ri) = coerce_ints(left, right, &self.builder);
                let result = self
                    .builder
                    .build_int_compare(IntPredicate::UGT, li, ri, &name)
                    .map_err(|e| format!("Failed to build ugt: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::UGe => {
                let (li, ri) = coerce_ints(left, right, &self.builder);
                let result = self
                    .builder
                    .build_int_compare(IntPredicate::UGE, li, ri, &name)
                    .map_err(|e| format!("Failed to build uge: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }

            // Float comparisons (ordered)
            CompareOp::FEq => {
                let result = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::OEQ,
                        left.into_float_value(),
                        right.into_float_value(),
                        &name,
                    )
                    .map_err(|e| format!("Failed to build feq: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::FNe => {
                let result = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::ONE,
                        left.into_float_value(),
                        right.into_float_value(),
                        &name,
                    )
                    .map_err(|e| format!("Failed to build fne: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::FLt => {
                let result = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::OLT,
                        left.into_float_value(),
                        right.into_float_value(),
                        &name,
                    )
                    .map_err(|e| format!("Failed to build flt: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::FLe => {
                let result = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::OLE,
                        left.into_float_value(),
                        right.into_float_value(),
                        &name,
                    )
                    .map_err(|e| format!("Failed to build fle: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::FGt => {
                let result = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::OGT,
                        left.into_float_value(),
                        right.into_float_value(),
                        &name,
                    )
                    .map_err(|e| format!("Failed to build fgt: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::FGe => {
                let result = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::OGE,
                        left.into_float_value(),
                        right.into_float_value(),
                        &name,
                    )
                    .map_err(|e| format!("Failed to build fge: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }

            // Ordered/Unordered comparisons
            CompareOp::FOrd => {
                let result = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::ORD,
                        left.into_float_value(),
                        right.into_float_value(),
                        &name,
                    )
                    .map_err(|e| format!("Failed to build ford: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
            CompareOp::FUno => {
                let result = self
                    .builder
                    .build_float_compare(
                        FloatPredicate::UNO,
                        left.into_float_value(),
                        right.into_float_value(),
                        &name,
                    )
                    .map_err(|e| format!("Failed to build funo: {}", e))?;
                to_i8(result, &self.builder, &format!("{}_i8", name))
            }
        }
    }

    /// Compile a direct function call
    fn compile_direct_call(
        &mut self,
        func_id: IrFunctionId,
        args: &[IrId],
    ) -> Result<Option<BasicValueEnum<'ctx>>, String> {
        let llvm_func = self
            .function_map
            .get(&func_id)
            .ok_or_else(|| format!("Function {:?} not found", func_id))?;

        // Get expected parameter types from the function
        let expected_params = llvm_func.get_type().get_param_types();

        // Determine calling convention by examining LLVM function signature
        // This is more robust than tracking by func_id since IDs can differ across modules
        //
        // Calling convention patterns:
        // - C extern: expected_params.len() == args.len() (no hidden params)
        // - Haxe no sret: expected_params.len() == args.len() + 1, first param is i64 (env)
        // - Haxe with sret: expected_params.len() == args.len() + 2, first is ptr, second is i64
        let num_llvm_params = expected_params.len();
        let num_ir_args = args.len();

        // Check first param is i64 (env pattern)
        let first_is_i64 = expected_params
            .first()
            .map(|p| p.is_int_type() && p.into_int_type().get_bit_width() == 64)
            .unwrap_or(false);

        // Check first is ptr and second is i64 (sret + env pattern)
        let first_is_ptr = expected_params
            .first()
            .map(|p| p.is_pointer_type())
            .unwrap_or(false);
        let second_is_i64 = expected_params
            .get(1)
            .map(|p| p.is_int_type() && p.into_int_type().get_bit_width() == 64)
            .unwrap_or(false);

        // Determine convention based on parameter count and signature patterns
        let (uses_sret, expects_env) = if num_llvm_params == num_ir_args {
            // Exact match - C calling convention (no hidden params)
            (false, false)
        } else if num_llvm_params == num_ir_args + 1 && first_is_i64 {
            // One extra param that's i64 - Haxe convention with env only
            (false, true)
        } else if num_llvm_params == num_ir_args + 2 && first_is_ptr && second_is_i64 {
            // Two extra params (ptr + i64) - Haxe convention with sret + env
            (true, true)
        } else {
            // Fallback: use tracked sret and assume env for non-extern
            let tracked_sret = self.sret_function_ids.contains(&func_id);
            let is_extern = self.extern_function_ids.contains(&func_id);
            (tracked_sret && !is_extern, !is_extern)
        };

        // Get argument values and coerce them to match expected types
        let mut arg_values: Vec<BasicMetadataValueEnum> = Vec::new();

        // Allocate sret stack space if needed and add as first argument
        let sret_slot = if uses_sret {
            // Get the return type from the function (stored in sret ptr type)
            // The sret pointer points to the return value type
            // We need to allocate stack space for it
            let sret_ptr_ty = expected_params
                .first()
                .ok_or("sret function missing sret parameter")?;

            // Allocate stack space for the return value
            // Use alloca with the pointee type (we need to determine the struct size)
            // For now, allocate a generic buffer; LLVM will optimize
            let alloca = self
                .builder
                .build_alloca(
                    self.context.i8_type().array_type(64), // 64 bytes should be enough for most structs
                    "sret_slot",
                )
                .map_err(|e| format!("Failed to allocate sret slot: {}", e))?;

            // Cast to the expected pointer type if needed
            arg_values.push(alloca.into());
            Some(alloca)
        } else {
            None
        };

        // Add hidden env parameter (null/0) only if function expects it
        let param_offset = if expects_env {
            arg_values.push(self.context.i64_type().const_int(0, false).into());
            if uses_sret {
                2
            } else {
                1
            } // sret + env, or just env
        } else if uses_sret {
            1 // just sret, no env
        } else {
            0 // No hidden params
        };

        for (i, &id) in args.iter().enumerate() {
            let val = self.get_value(id)?;

            // Coerce to expected parameter type if needed
            let coerced = if let Some(expected_ty) = expected_params.get(i + param_offset) {
                let actual_ty = val.get_type();
                if actual_ty == *expected_ty {
                    val.into()
                } else {
                    let cast_name = format!("arg_cast_{}", i);

                    // Handle struct -> ptr coercion (e.g., {i64, ptr} -> ptr)
                    if val.is_struct_value() && expected_ty.is_pointer_type() {
                        let struct_val = val.into_struct_value();
                        // Check the struct type to decide how to coerce:
                        // - {ptr, i64} = string struct → alloca+store (pass by reference)
                        // - {i64, ptr} = object wrapper → extract ptr field (index 1)
                        let struct_ty = struct_val.get_type();
                        let first_field_is_ptr = struct_ty.count_fields() >= 2
                            && struct_ty
                                .get_field_type_at_index(0)
                                .map(|t| t.is_pointer_type())
                                .unwrap_or(false);

                        if first_field_is_ptr {
                            // String-like struct {ptr, i64} — pass pointer to struct
                            let alloca = self
                                .builder
                                .build_alloca(struct_ty, &cast_name)
                                .map_err(|e| format!("Failed to alloca for struct->ptr: {}", e))?;
                            self.builder
                                .build_store(alloca, struct_val)
                                .map_err(|e| format!("Failed to store struct for ptr: {}", e))?;
                            alloca.into()
                        } else {
                            // Object wrapper {i64, ptr} — extract the pointer field
                            self.builder
                                .build_extract_value(struct_val, 1, &cast_name)
                                .map_err(|e| format!("Failed to extract ptr from struct: {}", e))?
                                .into()
                        }
                    } else if val.is_struct_value() && expected_ty.is_int_type() {
                        // Extract i64 from struct (field 0 is the length in {i64, ptr})
                        let struct_val = val.into_struct_value();
                        let extracted = self
                            .builder
                            .build_extract_value(struct_val, 0, &cast_name)
                            .map_err(|e| format!("Failed to extract i64 from struct: {}", e))?;

                        // May need to truncate/extend if target int type differs
                        let extracted_int = extracted.into_int_value();
                        let target_int_ty = expected_ty.into_int_type();
                        if extracted_int.get_type().get_bit_width() != target_int_ty.get_bit_width()
                        {
                            if extracted_int.get_type().get_bit_width()
                                > target_int_ty.get_bit_width()
                            {
                                self.builder
                                    .build_int_truncate(
                                        extracted_int,
                                        target_int_ty,
                                        &format!("{}_trunc", cast_name),
                                    )
                                    .map_err(|e| format!("Failed to truncate int: {}", e))?
                                    .into()
                            } else {
                                self.builder
                                    .build_int_s_extend(
                                        extracted_int,
                                        target_int_ty,
                                        &format!("{}_sext", cast_name),
                                    )
                                    .map_err(|e| format!("Failed to extend int: {}", e))?
                                    .into()
                            }
                        } else {
                            extracted.into()
                        }
                    } else if val.is_pointer_value() && expected_ty.is_struct_type() {
                        // Wrap ptr in struct (e.g., ptr -> {i64, ptr})
                        let struct_ty = expected_ty.into_struct_type();
                        let len_val = self.context.i64_type().const_int(0, false);
                        let ptr_val = val.into_pointer_value();
                        let s1 = self
                            .builder
                            .build_insert_value(
                                struct_ty.const_zero(),
                                len_val,
                                0,
                                &format!("{}_len", cast_name),
                            )
                            .map_err(|e| format!("Failed to wrap ptr in struct: {}", e))?
                            .into_struct_value();
                        let s2 = self
                            .builder
                            .build_insert_value(s1, ptr_val, 1, &format!("{}_ptr", cast_name))
                            .map_err(|e| format!("Failed to wrap ptr in struct: {}", e))?
                            .into_struct_value();
                        s2.into()
                    } else if val.is_float_value() && expected_ty.is_int_type() {
                        // Float to int
                        self.builder
                            .build_float_to_signed_int(
                                val.into_float_value(),
                                expected_ty.into_int_type(),
                                &cast_name,
                            )
                            .map_err(|e| format!("Failed to cast arg: {}", e))?
                            .into()
                    } else if val.is_int_value() && expected_ty.is_float_type() {
                        // Int to float
                        self.builder
                            .build_signed_int_to_float(
                                val.into_int_value(),
                                expected_ty.into_float_type(),
                                &cast_name,
                            )
                            .map_err(|e| format!("Failed to cast arg: {}", e))?
                            .into()
                    } else if val.is_int_value() && expected_ty.is_int_type() {
                        // Int to int with different widths
                        let int_val = val.into_int_value();
                        let target_int_ty = expected_ty.into_int_type();
                        if int_val.get_type().get_bit_width() < target_int_ty.get_bit_width() {
                            // Extend (sign extend for safety)
                            self.builder
                                .build_int_s_extend(int_val, target_int_ty, &cast_name)
                                .map_err(|e| format!("Failed to extend int: {}", e))?
                                .into()
                        } else if int_val.get_type().get_bit_width() > target_int_ty.get_bit_width()
                        {
                            // Truncate
                            self.builder
                                .build_int_truncate(int_val, target_int_ty, &cast_name)
                                .map_err(|e| format!("Failed to truncate int: {}", e))?
                                .into()
                        } else {
                            // Same width - use as-is
                            val.into()
                        }
                    } else if val.is_pointer_value() && expected_ty.is_int_type() {
                        // Ptr to int - convert pointer to integer
                        let ptr_val = val.into_pointer_value();
                        let target_int_ty = expected_ty.into_int_type();
                        self.builder
                            .build_ptr_to_int(ptr_val, target_int_ty, &cast_name)
                            .map_err(|e| format!("Failed to convert ptr to int: {}", e))?
                            .into()
                    } else if val.is_int_value() && expected_ty.is_pointer_type() {
                        // Int to ptr - convert integer to pointer
                        let int_val = val.into_int_value();
                        let target_ptr_ty = expected_ty.into_pointer_type();
                        self.builder
                            .build_int_to_ptr(int_val, target_ptr_ty, &cast_name)
                            .map_err(|e| format!("Failed to convert int to ptr: {}", e))?
                            .into()
                    } else {
                        // Use as-is
                        val.into()
                    }
                }
            } else {
                val.into()
            };

            arg_values.push(coerced);
        }

        let call_site = self
            .builder
            .build_call(*llvm_func, &arg_values, "call")
            .map_err(|e| format!("Failed to build call: {}", e))?;

        // For sret functions, the return value is in the sret slot, not the call result
        if let Some(sret_ptr) = sret_slot {
            // Load the return value from the sret slot
            // The sret slot contains the struct value written by the callee
            // Return the pointer to the sret slot as the "return value"
            // (Most uses will just use this pointer directly)
            Ok(Some(sret_ptr.into()))
        } else {
            Ok(call_site.try_as_basic_value().left())
        }
    }

    /// Compile type cast
    fn compile_cast(
        &self,
        src: BasicValueEnum<'ctx>,
        from_ty: &IrType,
        to_ty: &IrType,
        dest: IrId,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let name = format!("cast_{}", dest.as_u32());
        let target_llvm_ty = self.translate_type(to_ty)?;

        // Handle mismatch between MIR types and actual LLVM values
        // This can happen due to Haxe's numeric promotion (e.g., Int used as Float)
        let actual_is_float = src.is_float_value();
        let actual_is_int = src.is_int_value();

        // Handle type mismatches - when actual LLVM value type differs from MIR type
        // This happens due to Haxe's dynamic numeric promotion

        // If actual value is float but target is int, convert float->int
        if actual_is_float && to_ty.is_integer() {
            let target_int_ty = target_llvm_ty.into_int_type();
            let result = if to_ty.is_signed_integer() {
                self.builder
                    .build_float_to_signed_int(src.into_float_value(), target_int_ty, &name)
            } else {
                self.builder.build_float_to_unsigned_int(
                    src.into_float_value(),
                    target_int_ty,
                    &name,
                )
            }
            .map_err(|e| format!("Failed to build float-to-int cast: {}", e))?;
            return Ok(result.into());
        }

        // If actual value is float and target is float, do float-to-float
        if actual_is_float && to_ty.is_float() {
            let src_float = src.into_float_value();
            let target_float_ty = target_llvm_ty.into_float_type();
            // Just build the cast regardless of source size (LLVM will handle it)
            let result = if src_float.get_type() == target_float_ty {
                // Same type, just return
                return Ok(src);
            } else if src_float.get_type().get_context().f32_type() == src_float.get_type() {
                // f32 -> f64, extend
                self.builder
                    .build_float_ext(src_float, target_float_ty, &name)
            } else {
                // f64 -> f32 (or same), truncate or extend
                self.builder
                    .build_float_trunc(src_float, target_float_ty, &name)
            }
            .map_err(|e| format!("Failed to build float-to-float cast: {}", e))?;
            return Ok(result.into());
        }

        // If actual value is int but target is float, convert int->float
        if actual_is_int && to_ty.is_float() {
            let target_float_ty = target_llvm_ty.into_float_type();
            let result = self
                .builder
                .build_signed_int_to_float(src.into_int_value(), target_float_ty, &name)
                .map_err(|e| format!("Failed to build int-to-float cast: {}", e))?;
            return Ok(result.into());
        }

        // Integer to integer casts (normal path)
        if from_ty.is_integer() && to_ty.is_integer() {
            let src_int = src.into_int_value();
            let target_int_ty = target_llvm_ty.into_int_type();

            let result = if from_ty.size() < to_ty.size() {
                // Extend
                if from_ty.is_signed_integer() {
                    self.builder
                        .build_int_s_extend(src_int, target_int_ty, &name)
                } else {
                    self.builder
                        .build_int_z_extend(src_int, target_int_ty, &name)
                }
            } else {
                // Truncate
                self.builder
                    .build_int_truncate(src_int, target_int_ty, &name)
            }
            .map_err(|e| format!("Failed to build int cast: {}", e))?;

            return Ok(result.into());
        }

        // Float to float casts
        if from_ty.is_float() && to_ty.is_float() {
            let src_float = src.into_float_value();
            let target_float_ty = target_llvm_ty.into_float_type();

            let result = if from_ty.size() < to_ty.size() {
                self.builder
                    .build_float_ext(src_float, target_float_ty, &name)
            } else {
                self.builder
                    .build_float_trunc(src_float, target_float_ty, &name)
            }
            .map_err(|e| format!("Failed to build float cast: {}", e))?;

            return Ok(result.into());
        }

        // Int to float
        if from_ty.is_integer() && to_ty.is_float() {
            let src_int = src.into_int_value();
            let target_float_ty = target_llvm_ty.into_float_type();

            let result = if from_ty.is_signed_integer() {
                self.builder
                    .build_signed_int_to_float(src_int, target_float_ty, &name)
            } else {
                self.builder
                    .build_unsigned_int_to_float(src_int, target_float_ty, &name)
            }
            .map_err(|e| format!("Failed to build int to float: {}", e))?;

            return Ok(result.into());
        }

        // Float to int
        if from_ty.is_float() && to_ty.is_integer() {
            let src_float = src.into_float_value();
            let target_int_ty = target_llvm_ty.into_int_type();

            let result = if to_ty.is_signed_integer() {
                self.builder
                    .build_float_to_signed_int(src_float, target_int_ty, &name)
            } else {
                self.builder
                    .build_float_to_unsigned_int(src_float, target_int_ty, &name)
            }
            .map_err(|e| format!("Failed to build float to int: {}", e))?;

            return Ok(result.into());
        }

        // Pointer casts
        if from_ty.is_pointer() && to_ty.is_pointer() {
            let src_ptr = src.into_pointer_value();
            let target_ptr_ty = target_llvm_ty.into_pointer_type();

            let result = self
                .builder
                .build_pointer_cast(src_ptr, target_ptr_ty, &name)
                .map_err(|e| format!("Failed to build pointer cast: {}", e))?;

            return Ok(result.into());
        }

        // Pointer to integer
        if from_ty.is_pointer() && to_ty.is_integer() {
            let src_ptr = src.into_pointer_value();
            let target_int_ty = target_llvm_ty.into_int_type();

            let result = self
                .builder
                .build_ptr_to_int(src_ptr, target_int_ty, &name)
                .map_err(|e| format!("Failed to build ptr to int: {}", e))?;

            return Ok(result.into());
        }

        // Integer to pointer
        if from_ty.is_integer() && to_ty.is_pointer() {
            let src_int = src.into_int_value();
            let target_ptr_ty = target_llvm_ty.into_pointer_type();

            let result = self
                .builder
                .build_int_to_ptr(src_int, target_ptr_ty, &name)
                .map_err(|e| format!("Failed to build int to ptr: {}", e))?;

            return Ok(result.into());
        }

        // AOT: struct to pointer cast
        if self.aot_mode && src.is_struct_value() && to_ty.is_pointer() {
            let struct_val = src.into_struct_value();
            let struct_ty = struct_val.get_type();
            let first_field_is_ptr = struct_ty.count_fields() >= 2
                && struct_ty
                    .get_field_type_at_index(0)
                    .map(|t| t.is_pointer_type())
                    .unwrap_or(false);

            if first_field_is_ptr {
                // String-like struct {ptr, i64} → alloca+store, pass by reference
                let alloca = self
                    .builder
                    .build_alloca(struct_ty, &name)
                    .map_err(|e| format!("Failed to alloca for struct->ptr cast: {}", e))?;
                self.builder
                    .build_store(alloca, struct_val)
                    .map_err(|e| format!("Failed to store struct for ptr cast: {}", e))?;
                return Ok(alloca.into());
            } else {
                // Object wrapper {i64, ptr} → extract the pointer field
                let extracted = self
                    .builder
                    .build_extract_value(struct_val, 1, &name)
                    .map_err(|e| format!("Failed to extract ptr from struct cast: {}", e))?;
                return Ok(extracted);
            }
        }

        // Any type to integer - Any is now ptr, so we need ptrtoint
        if matches!(from_ty, IrType::Any) && to_ty.is_integer() {
            let target_int_ty = target_llvm_ty.into_int_type();
            if src.is_pointer_value() {
                // Any (ptr) to integer - use ptrtoint
                let result = self
                    .builder
                    .build_ptr_to_int(src.into_pointer_value(), target_int_ty, &name)
                    .map_err(|e| format!("Failed to build ptr to int from Any: {}", e))?;
                return Ok(result.into());
            } else if src.is_int_value() {
                // Already int (shouldn't happen if Any is ptr, but handle it)
                let src_int = src.into_int_value();
                if src_int.get_type() == target_int_ty {
                    return Ok(src);
                }
                let result = self
                    .builder
                    .build_int_z_extend_or_bit_cast(src_int, target_int_ty, &name)
                    .map_err(|e| format!("Failed to cast int from Any: {}", e))?;
                return Ok(result.into());
            }
        }

        // Any type to pointer - Any is already ptr, just cast
        if matches!(from_ty, IrType::Any) && to_ty.is_pointer() {
            let target_ptr_ty = target_llvm_ty.into_pointer_type();
            if src.is_pointer_value() {
                // Already a pointer - just cast (no conversion needed with opaque ptrs)
                let result = self
                    .builder
                    .build_pointer_cast(src.into_pointer_value(), target_ptr_ty, &name)
                    .map_err(|e| format!("Failed to cast ptr from Any: {}", e))?;
                return Ok(result.into());
            } else if src.is_int_value() {
                // Int to ptr (shouldn't happen if Any is ptr, but handle it)
                let result = self
                    .builder
                    .build_int_to_ptr(src.into_int_value(), target_ptr_ty, &name)
                    .map_err(|e| format!("Failed to build int to ptr from Any: {}", e))?;
                return Ok(result.into());
            }
        }

        // Pointer to Any - Any is ptr, just cast (no conversion needed)
        if from_ty.is_pointer() && matches!(to_ty, IrType::Any) {
            let target_ptr_ty = self.context.ptr_type(AddressSpace::default());
            if src.is_pointer_value() {
                let result = self
                    .builder
                    .build_pointer_cast(src.into_pointer_value(), target_ptr_ty, &name)
                    .map_err(|e| format!("Failed to cast ptr to Any: {}", e))?;
                return Ok(result.into());
            }
        }

        // Integer to Any - Any is ptr, use inttoptr
        if from_ty.is_integer() && matches!(to_ty, IrType::Any) {
            let target_ptr_ty = self.context.ptr_type(AddressSpace::default());
            if src.is_int_value() {
                let result = self
                    .builder
                    .build_int_to_ptr(src.into_int_value(), target_ptr_ty, &name)
                    .map_err(|e| format!("Failed to build int to ptr for Any: {}", e))?;
                return Ok(result.into());
            }
        }

        Err(format!(
            "Unsupported cast from {:?} to {:?}",
            from_ty, to_ty
        ))
    }

    /// Get the size in bytes of an IR type.
    /// Used for GEP element size calculation.
    fn ir_type_size(ty: &crate::ir::IrType) -> u64 {
        use crate::ir::IrType;
        match ty {
            IrType::I8 | IrType::U8 | IrType::Bool => 1,
            IrType::I16 | IrType::U16 => 2,
            IrType::I32 | IrType::U32 | IrType::F32 => 4,
            IrType::I64 | IrType::U64 | IrType::F64 => 8,
            IrType::Ptr(_) | IrType::Ref(_) => 8, // 64-bit pointers
            IrType::Void => 0,
            IrType::Any => 8,    // Boxed value pointer
            IrType::String => 8, // String is a pointer
            IrType::Vector { element, count } => Self::ir_type_size(element) * (*count as u64),
            _ => 8, // Default to pointer size
        }
    }
}

// Stub implementation when LLVM backend is disabled
#[cfg(not(feature = "llvm-backend"))]
pub struct LLVMJitBackend {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "llvm-backend"))]
impl LLVMJitBackend {
    pub fn new(_context: &()) -> Result<Self, String> {
        Err("LLVM backend not enabled. Compile with --features llvm-backend".to_string())
    }

    pub fn compile_single_function(
        &mut self,
        _func_id: crate::ir::IrFunctionId,
        _function: &crate::ir::IrFunction,
    ) -> Result<(), String> {
        Err("LLVM backend not enabled".to_string())
    }

    pub fn get_function_ptr(&self, _func_id: crate::ir::IrFunctionId) -> Result<*const u8, String> {
        Err("LLVM backend not enabled".to_string())
    }
}
