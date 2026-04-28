//! Haxe Macro System
//!
//! This module implements compile-time macro expansion for the Rayzor compiler.
//! Macros are Haxe functions marked with the `macro` modifier that execute at
//! compile time, transforming AST expressions before type checking and code generation.
//!
//! # Architecture
//!
//! ```text
//! Source Code
//!     │
//!     ▼
//!   Parse → AST (HaxeFile)
//!     │
//!     ├─ registry: scan for `macro` functions, @:build metadata
//!     ├─ class_registry: scan for class declarations (constructors, methods)
//!     │
//!     ▼
//!   Macro Expansion (expander.rs)
//!     │
//!     ├─ interpreter.rs (tree-walker) ◄─── cold macros
//!     ├─ bytecode/vm.rs (bytecode VM) ◄─── hot macros (promoted by MorselScheduler)
//!     ├─ context_api.rs (Context.* methods)
//!     ├─ reification.rs ($v{}, $i{}, $e{}, etc.)
//!     ├─ build_macros.rs (@:build, @:autoBuild)
//!     │
//!     ▼
//!   Expanded AST → TAST → HIR → MIR → Codegen
//! ```
//!
//! # Execution Model
//!
//! The interpreter uses a **tiered execution** strategy inspired by morsel-parallelism:
//!
//! - **Cold macros** (called fewer than threshold times): executed by the tree-walking
//!   interpreter (`interpreter.rs`). Zero compilation overhead.
//! - **Hot macros** (called ≥ threshold times): promoted to bytecode by the
//!   `MorselScheduler`. The macro body + all class dependencies are batch-compiled
//!   to bytecode ("morsel"), then executed by the stack-based VM (`bytecode/vm.rs`).
//!
//! This tiering is transparent — the same `call_macro_def()` entry point handles both
//! paths. Enable via `RAYZOR_MACRO_VM=1`; tune threshold via `RAYZOR_MACRO_VM_THRESHOLD`.
//!
//! # Submodules
//!
//! - **`interpreter`**: Tree-walking evaluator (~2400 lines, 35+ ExprKind arms)
//! - **`bytecode`**: Bytecode VM (62 opcodes, compiler, disassembler)
//! - **`expander`**: Orchestrates macro expansion pipeline
//! - **`registry`**: Tracks macro definitions and compiled bytecode chunks
//! - **`class_registry`**: Tracks class declarations for constructor/method dispatch
//! - **`reification`**: Bidirectional AST ↔ macro value conversion
//! - **`context_api`**: `haxe.macro.Context` implementation
//! - **`build_macros`**: `@:build` / `@:autoBuild` metadata processing
//! - **`environment`**: Lexical scoping with scope stack
//! - **`value`**: Runtime value types (MacroValue enum, Arc-based COW)
//! - **`ast_bridge`**: Conversion between parser AST and macro values
//! - **`errors`**: Macro-specific error types and diagnostics

pub mod ast_bridge;
pub mod build_macros;
pub mod bytecode;
pub mod class_registry;
pub mod context_api;
pub mod environment;
pub mod errors;
pub mod expander;
pub mod interpreter;
pub mod registry;
pub mod reification;
pub mod value;

pub use ast_bridge::{apply_binary_op, expr_to_value, value_to_expr};
pub use build_macros::{
    process_build_macros, process_build_macros_with_class_registry, BuildMacroResult,
};
pub use class_registry::ClassRegistry;
pub use context_api::{
    BuildClassContext, BuildField, BuildFieldKind, DefinedType, DefinedTypeKind, FieldAccess,
    FieldMeta, MacroContext,
};
pub use environment::Environment;
pub use errors::{MacroDiagnostic, MacroError, MacroSeverity, PipelineDiagnostic};
pub use expander::{
    expand_macros, expand_macros_with_class_registry, expand_macros_with_dependencies,
    expand_macros_with_registry, ExpansionResult, MacroExpander,
};
pub use interpreter::MacroInterpreter;
pub use registry::{BuildMacroEntry, MacroDefinition, MacroRegistry};
pub use reification::ReificationEngine;
pub use value::{MacroFunction, MacroParam, MacroValue};
