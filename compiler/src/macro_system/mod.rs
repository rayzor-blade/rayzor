//! Haxe Macro System
//!
//! This module implements compile-time macro expansion for the Rayzor compiler.
//! It covers:
//!
//! - **Macro Interpreter**: Tree-walking interpreter for executing macro function
//!   bodies at compile time
//! - **Reification Engine**: Bidirectional conversion between AST nodes and macro
//!   values ($v{}, $i{}, $e{}, $a{}, $p{}, $b{})
//! - **Context API**: Implementation of `haxe.macro.Context` methods
//! - **Build Macros**: `@:build` and `@:autoBuild` metadata processing
//! - **Pipeline Integration**: Macro expansion stages between parsing and TAST lowering

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
pub use build_macros::{process_build_macros, BuildMacroResult};
pub use class_registry::ClassRegistry;
pub use context_api::{
    BuildClassContext, BuildField, BuildFieldKind, DefinedType, DefinedTypeKind, FieldAccess,
    FieldMeta, MacroContext,
};
pub use environment::Environment;
pub use errors::{MacroDiagnostic, MacroError, MacroSeverity, PipelineDiagnostic};
pub use expander::{
    expand_macros, expand_macros_with_class_registry, expand_macros_with_registry, ExpansionResult,
    MacroExpander,
};
pub use interpreter::MacroInterpreter;
pub use registry::{BuildMacroEntry, MacroDefinition, MacroRegistry};
pub use reification::ReificationEngine;
pub use value::{MacroFunction, MacroParam, MacroValue};
