//! Intermediate Representation (IR) for the Haxe Compiler
//!
//! This module defines a low-level, platform-independent intermediate representation
//! that serves as the target for TAST lowering and the source for code generation.
//! The IR is designed to be:
//! - Simple and explicit (no implicit operations)
//! - Strongly typed with explicit type information
//! - Easy to optimize and transform
//! - Suitable for targeting multiple backends (JS, C++, JVM, etc.)

pub mod drop_analysis;
pub mod hir; // High-level IR (close to source syntax)
pub mod hir_to_mir; // HIR to MIR lowering
pub mod tast_to_hir; // TAST to HIR lowering // Drop point analysis for automatic memory deallocation

// MIR modules (the existing IR serves as MIR)
pub mod blade; // BLADE format - Blazing Language Artifact Deployment Environment (.blade files)
pub mod blocks;
pub mod bounds_check_elimination; // Bounds Check Elimination for array loops
pub mod builder;
pub mod devirtualize; // Devirtualization: indirect → direct calls
pub mod dump; // MIR pretty-printer for debugging
pub mod environment_layout; // Closure environment layout abstraction
pub mod escape_analysis; // Intra-loop escape analysis for Alloc hoisting
pub mod functions;
pub mod inlining; // Function inlining and call graph analysis
pub mod insert_free; // Insert Free instructions for non-escaping allocations
pub mod instructions;
pub mod loop_analysis; // Loop analysis: dominators, natural loops, nesting
pub mod loop_unrolling; // Loop unrolling for constant-trip-count loops
pub mod lowering; // Legacy TAST to MIR (being phased out)
pub mod mir_builder; // Programmatic MIR construction API
pub mod modules;
pub mod monomorphize; // Monomorphization pass for generics
pub mod optimizable; // Generic optimization trait for different IR levels
pub mod optimization;
pub mod scalar_replacement; // Scalar Replacement of Aggregates (SRA)
pub mod tree_shake; // Dead-code elimination for .rzb bundles
pub mod types;
pub mod validation;
pub mod vectorization; // SIMD auto-vectorization for loops

pub use blade::{load_bundle, save_bundle, BladeError, RayzorBundle};
pub use blocks::*;
pub use builder::*;
pub use environment_layout::{EnvironmentField, EnvironmentLayout};
pub use functions::*;
pub use instructions::*;
pub use loop_analysis::{DominatorTree, LoopNestInfo, NaturalLoop, TripCount};
pub use modules::*;
pub use monomorphize::{MonoKey, MonomorphizationStats, Monomorphizer};
pub use types::*;
pub use vectorization::{LoopVectorizationPass, VectorInstruction, VectorType};

use serde::{Deserialize, Serialize};
use std::fmt;

/// IR version for compatibility checking
pub const IR_VERSION: u32 = 1;

/// Unique identifier for IR entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IrId(u32);

impl IrId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn invalid() -> Self {
        Self(u32::MAX)
    }

    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for IrId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.0)
    }
}

/// Source location information for debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrSourceLocation {
    pub file_id: u32,
    pub line: u32,
    pub column: u32,
}

impl IrSourceLocation {
    pub fn unknown() -> Self {
        Self {
            file_id: 0,
            line: 0,
            column: 0,
        }
    }
}

/// Linkage type for symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Linkage {
    /// Private to the module
    Private,
    /// Available within the package
    Internal,
    /// Publicly exported
    Public,
    /// External symbol (defined elsewhere)
    External,
}

/// Calling convention for functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallingConvention {
    /// Standard Haxe calling convention
    Haxe,
    /// C calling convention (for FFI)
    C,
    /// Fast calling convention (optimized)
    Fast,
    /// Platform-specific convention
    Native,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_id() {
        let id = IrId::new(42);
        assert_eq!(format!("{}", id), "$42");
        assert!(id.is_valid());

        let invalid = IrId::invalid();
        assert!(!invalid.is_valid());
    }
}
