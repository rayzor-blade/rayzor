pub mod ast_lowering;
pub mod capture_analyzer;
pub mod class_builder;
pub mod constraint_solver;
pub mod control_flow_analysis;
pub mod core;
pub mod core_types;
pub mod effect_analysis;
pub mod generic_instantiation;
pub mod generics;
pub mod id_types;
pub mod namespace;
pub mod node;
pub mod node_extensions;
pub mod null_safety_analysis;
pub mod package_access;
pub mod scopes;
pub mod send_sync_validator;
pub mod source_extractor;
pub mod span_conversion;
pub mod stdlib_loader;
pub mod string_intern;
pub mod symbol_cache;
pub mod symbols;
pub mod trait_checker;
pub mod type_arena;
pub mod type_cache;
pub mod type_checker;
pub mod type_checking_pipeline;
pub mod type_diagnostics;
pub mod type_flow_guard;
pub mod type_resolution;
pub mod type_resolution_helpers;
// pub mod type_inference;

#[cfg(test)]
mod tests;

pub use ast_lowering::*;
pub use control_flow_analysis::VariableState as FlowVariableState;
pub use core::{Type, TypeFlags, TypeKind, TypeTable, Variance};
pub use id_types::*;
pub use namespace::{
    ImportEntry, ImportResolver, NamespaceResolver, PackageId, PackageInfo, QualifiedPath,
};
pub use node::{
    AsyncKind, DerivedTrait, FunctionEffects, MemoryAnnotation, MemoryEffects, PropertyAccessInfo,
    PropertyAccessor, ResourceEffects, SafetyMode, TypedClass, TypedEnum, TypedExpression,
    TypedExpressionKind, TypedFile, TypedInterface, TypedStatement,
};
pub use package_access::{AccessPermission, PackageAccessContext, PackageAccessValidator};
pub use scopes::*;
pub use string_intern::*;
pub use symbols::*;
pub use type_arena::*;
pub use type_checker::{AccessLevel, TypeCheckError, TypeCheckResult, TypeChecker, TypeErrorKind};
pub use type_flow_guard::{FlowAnalysisMetrics, FlowSafetyError, FlowSafetyResults, TypeFlowGuard};
// pub use type_resolution::*;
// pub use type_inference::*;
