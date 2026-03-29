/// Code generation backends for Rayzor
///
/// This module contains the code generation infrastructure for targeting
/// different backends:
/// - MIR Interpreter (instant startup, Phase 0)
/// - Cranelift (JIT with tiered compilation, Phases 1-3)
/// - LLVM (maximum optimization, Phase 4)
/// - WebAssembly (cross-platform AOT, browser + WASI)
pub mod aot_compiler;
pub mod c_backend;
pub mod cranelift_backend;
mod instruction_lowering;
pub mod llvm_aot_backend;
pub mod llvm_jit_backend;
pub mod mir_interpreter;
pub mod profiling;
pub mod tiered_backend;
pub mod wasm_backend;
pub mod wasm_linker;
pub mod wasm_runner;
pub mod wgsl_transpiler;

// Apple Silicon-specific JIT memory management
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod apple_jit_memory;

pub use cranelift_backend::CraneliftBackend;
pub use mir_interpreter::{
    DecodedBlock, DecodedInstruction, HeapObject, InterpError, InterpValue, MirInterpreter,
    NanBoxedValue, ObjectHeap, Opcode,
};
pub use profiling::{HotnessLevel, ProfileConfig, ProfileData, ProfileStatistics};
pub use tiered_backend::{
    BailoutStrategy, OptimizationTier, TierPreset, TieredBackend, TieredConfig, TieredStatistics,
};

#[cfg(feature = "llvm-backend")]
pub use llvm_jit_backend::{init_llvm_once, llvm_lock, reset_llvm_global_state, LLVMJitBackend};
