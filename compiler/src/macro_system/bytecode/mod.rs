//! Bytecode VM for the macro interpreter.
//!
//! Compiles Haxe macro function bodies from AST to flat bytecode, then
//! executes via a stack-based VM. Promoted macros get O(1) local variable
//! access and better cache locality compared to the tree-walking interpreter.
//!
//! # Architecture
//!
//! ```text
//! Expr (AST)                     BytecodeCompiler
//!   │                                  │
//!   ▼                                  ▼
//! compile() / compile_method()    Chunk {
//!                                   code: Vec<u8>,      ← flat bytecode
//!                                   constants: Vec<MacroValue>,
//!                                   local_count: u16,
//!                                   closures: Vec<Chunk>,
//!                                 }
//!                                      │
//!                                      ▼
//!                                 MacroVm::execute()
//!                                   stack: Vec<MacroValue>
//!                                   frames: Vec<CallFrame>
//! ```
//!
//! # Instruction Set
//!
//! 62 opcodes with variable-length encoding (1–5 bytes per instruction):
//! - **Stack**: Const, PushNull/True/False/Int0/Int1, Pop, Dup, Swap
//! - **Locals**: LoadLocal, StoreLocal, DefineLocal, LoadUpvalue
//! - **Arithmetic**: Add, Sub, Mul, Div, Mod, Neg, Incr, Decr
//! - **Comparison**: Eq, NotEq, Lt, Le, Gt, Ge
//! - **Bitwise**: BitAnd, BitOr, BitXor, Shl, Shr, Ushr, BitNot
//! - **Logic**: Not, NullCoal
//! - **Control**: Jump, JumpIfFalse/True, JumpIfFalseKeep/TrueKeep
//! - **Calls**: Call, CallMethod, CallStatic, CallBuiltin, CallMacroDef
//! - **Fields**: GetField, SetField, SetFieldLocal, GetFieldOpt, GetIndex, SetIndex
//! - **Construction**: MakeArray, MakeObject, MakeMap, MakeClosure, NewObject
//! - **Macro**: Reify, DollarSplice, MacroWrap
//! - **Control flow**: Return, ReturnNull
//!
//! # Submodules
//!
//! - `opcode`: Op enum, byte encoding/decoding, Emitter/Reader
//! - `chunk`: Chunk struct, CompiledParam, UpvalueDesc
//! - `compiler`: BytecodeCompiler (AST → Chunk)
//! - `vm`: MacroVm execution engine, CompiledClassInfo, call frames
//! - `debug`: Disassembler (human-readable bytecode dump)

pub mod chunk;
pub mod compiler;
pub mod debug;
pub mod opcode;
pub mod vm;

pub use chunk::{Chunk, CompiledParam, UpvalueDesc};
pub use compiler::{BytecodeCompiler, CompileError};
pub use opcode::{Emitter, Op, Reader};
pub use vm::{CompiledClassInfo, MacroVm, VmError};
