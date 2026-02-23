//! Bytecode VM for the macro interpreter.
//!
//! Compiles Haxe macro function bodies from AST to flat bytecode at
//! registration time, then executes via a stack-based VM. This replaces
//! the tree-walking interpreter for compiled macros, providing better
//! cache locality and O(1) local variable access.
//!
//! # Architecture
//!
//! - `opcode`: Instruction set (61 opcodes) with variable-length encoding
//! - `chunk`: Compilation unit (bytecode + constants pool + metadata)
//! - `compiler`: AST → Chunk compiler
//! - `vm`: Stack-based execution engine
//! - `builtins`: Built-in function dispatch table (planned)
//! - `debug`: Disassembler for debugging (planned)

pub mod chunk;
pub mod compiler;
pub mod debug;
pub mod opcode;
pub mod vm;

pub use chunk::{Chunk, CompiledParam, UpvalueDesc};
pub use compiler::{BytecodeCompiler, CompileError};
pub use opcode::{Emitter, Op, Reader};
pub use vm::{CompiledClassInfo, MacroVm, VmError};
