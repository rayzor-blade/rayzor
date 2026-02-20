//! HIR Functions
//!
//! This module defines function representation in the HIR, including
//! function signatures, parameters, local variables, and the function body.

use super::{
    CallingConvention, IrBlockId, IrControlFlowGraph, IrId, IrSourceLocation, IrType, Linkage,
};
use crate::tast::SymbolId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HIR function representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFunction {
    /// Unique identifier for this function
    pub id: IrFunctionId,

    /// Original symbol from TAST
    pub symbol_id: SymbolId,

    /// Function name (mangled if necessary)
    pub name: String,

    /// Fully qualified name (e.g., "com.example.MyClass.myMethod") for debugging and profiling
    pub qualified_name: Option<String>,

    /// Function signature
    pub signature: IrFunctionSignature,

    /// Control flow graph (function body)
    pub cfg: IrControlFlowGraph,

    /// Local variable declarations
    pub locals: HashMap<IrId, IrLocal>,

    /// Type information for all registers (parameters and intermediate values)
    /// This is populated by MirBuilder to enable type inference in backends
    pub register_types: HashMap<IrId, IrType>,

    /// Function attributes
    pub attributes: FunctionAttributes,

    /// Classification of function origin (user-defined, MIR wrapper, extern, etc.)
    pub kind: FunctionKind,

    /// Source location for debugging
    pub source_location: IrSourceLocation,

    /// Next available register ID (pub for MIR builder)
    pub next_reg_id: u32,

    /// Fixups for type parameter tag constants that need resolution during monomorphization.
    /// Each entry is (register_id_of_const_placeholder, type_param_name).
    /// The monomorphize pass replaces the placeholder const value with the correct type tag
    /// based on the concrete type that the type parameter resolves to.
    #[serde(default)]
    pub type_param_tag_fixups: Vec<(IrId, String)>,
}

/// Unique identifier for functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IrFunctionId(pub u32);

impl std::fmt::Display for IrFunctionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn{}", self.0)
    }
}

/// Function signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFunctionSignature {
    /// Parameter types and names
    pub parameters: Vec<IrParameter>,

    /// Return type
    pub return_type: IrType,

    /// Calling convention
    pub calling_convention: CallingConvention,

    /// Whether this function can throw
    pub can_throw: bool,

    /// Generic type parameters (if any)
    pub type_params: Vec<IrTypeParam>,

    /// Whether this function uses sret (structure return) convention
    /// When true, caller allocates space for return value and passes pointer as first param
    pub uses_sret: bool,
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrParameter {
    /// Parameter name
    pub name: String,

    /// Parameter type
    pub ty: IrType,

    /// Register assigned to this parameter
    pub reg: IrId,

    /// Whether this parameter is passed by reference
    pub by_ref: bool,
}

/// Generic type parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTypeParam {
    /// Type parameter name
    pub name: String,

    /// Constraints on this type parameter
    pub constraints: Vec<String>,
}

/// Local variable declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrLocal {
    /// Variable name (for debugging)
    pub name: String,

    /// Variable type
    pub ty: IrType,

    /// Whether this is mutable
    pub mutable: bool,

    /// Source location
    pub source_location: IrSourceLocation,

    /// Allocation hint from escape analysis
    pub allocation: AllocationHint,
}

/// Allocation hint for local variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationHint {
    /// Allocate on stack
    Stack,

    /// Allocate on heap
    Heap,

    /// Keep in register if possible
    Register,

    /// Compiler decides
    Auto,
}

/// Function attributes and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionAttributes {
    /// Linkage type
    pub linkage: Linkage,

    /// Whether this function is inline
    pub inline: InlineHint,

    /// Whether this function is pure (no side effects)
    pub pure: bool,

    /// Whether this function never returns
    pub no_return: bool,

    /// Whether this function should be optimized for size
    pub optimize_size: bool,

    /// Custom attributes
    pub custom: HashMap<String, String>,
}

/// Inline hint for functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InlineHint {
    /// Never inline
    Never,

    /// Compiler decides
    Auto,

    /// Prefer to inline
    Hint,

    /// Always inline
    Always,
}

/// Classification of function origin and calling convention.
///
/// This enum explicitly identifies where a function comes from and how it should
/// be handled during compilation, replacing fragile name-based detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FunctionKind {
    /// User-defined function from Haxe source code
    #[default]
    UserDefined,
    /// MIR wrapper function from stdlib (compiled by Cranelift, uses Haxe ABI)
    /// Examples: Thread_spawn, Channel_send, VecI32_push
    MirWrapper,
    /// Extern C function declaration (linked at JIT time, uses C ABI)
    /// Examples: haxe_string_char_at, rayzor_thread_spawn
    ExternC,
    /// Compiler intrinsic (special handling in codegen)
    Intrinsic,
}

impl Default for FunctionAttributes {
    fn default() -> Self {
        Self {
            linkage: Linkage::Private,
            inline: InlineHint::Auto,
            pure: false,
            no_return: false,
            optimize_size: false,
            custom: HashMap::new(),
        }
    }
}

impl IrFunction {
    /// Create a new HIR function
    pub fn new(
        id: IrFunctionId,
        symbol_id: SymbolId,
        name: String,
        signature: IrFunctionSignature,
    ) -> Self {
        let mut function = Self {
            id,
            symbol_id,
            name,
            qualified_name: None, // Set later during lowering
            signature,
            cfg: IrControlFlowGraph::new(),
            locals: HashMap::new(),
            register_types: HashMap::new(),
            attributes: FunctionAttributes::default(),
            kind: FunctionKind::UserDefined,
            source_location: IrSourceLocation::unknown(),
            next_reg_id: 0,
            type_param_tag_fixups: Vec::new(),
        };

        // Allocate registers for parameters and register their types
        let param_count = function.signature.parameters.len();
        for i in 0..param_count {
            let reg = function.alloc_reg();
            let param_ty = function.signature.parameters[i].ty.clone();
            function.signature.parameters[i].reg = reg;
            function.register_types.insert(reg, param_ty);
        }

        function
    }

    /// Allocate a new register
    pub fn alloc_reg(&mut self) -> IrId {
        let id = IrId::new(self.next_reg_id);
        self.next_reg_id += 1;
        id
    }

    /// Declare a local variable
    pub fn declare_local(&mut self, name: String, ty: IrType) -> IrId {
        let reg = self.alloc_reg();
        let local = IrLocal {
            name,
            ty,
            mutable: true,
            source_location: IrSourceLocation::unknown(),
            allocation: AllocationHint::Auto,
        };
        self.locals.insert(reg, local);
        reg
    }

    /// Get the entry block
    pub fn entry_block(&self) -> IrBlockId {
        self.cfg.entry_block
    }

    /// Get parameter register by index
    pub fn get_param_reg(&self, index: usize) -> Option<IrId> {
        self.signature.parameters.get(index).map(|p| p.reg)
    }

    /// Check if this function is a leaf function (doesn't call other functions)
    pub fn is_leaf(&self) -> bool {
        use super::IrInstruction;

        for block in self.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::CallDirect { .. } | IrInstruction::CallIndirect { .. } => {
                        return false
                    }
                    _ => {}
                }
            }
        }
        true
    }

    /// Get all registers used in this function
    pub fn used_registers(&self) -> Vec<IrId> {
        let mut regs = Vec::new();

        // Add parameter registers
        for param in &self.signature.parameters {
            regs.push(param.reg);
        }

        // Add local registers
        regs.extend(self.locals.keys().copied());

        regs
    }

    /// Verify function integrity
    pub fn verify(&self) -> Result<(), String> {
        // Skip verification for extern functions (no body/blocks)
        if self.cfg.blocks.is_empty() {
            return Ok(());
        }

        // Verify CFG
        self.cfg.verify()?;

        // Verify entry block has no phi nodes
        if let Some(entry) = self.cfg.get_block(self.cfg.entry_block) {
            if !entry.phi_nodes.is_empty() {
                return Err("Entry block cannot have phi nodes".to_string());
            }
        }

        // TODO: Add more verification (type checking, register usage, etc.)

        Ok(())
    }
}

/// Function optimization statistics
#[derive(Debug, Default)]
pub struct FunctionStats {
    /// Number of basic blocks
    pub block_count: usize,

    /// Number of instructions
    pub instruction_count: usize,

    /// Number of phi nodes
    pub phi_count: usize,

    /// Maximum block nesting depth
    pub max_depth: usize,

    /// Number of loops
    pub loop_count: usize,
}

impl IrFunction {
    /// Compute statistics for this function
    pub fn compute_stats(&self) -> FunctionStats {
        let mut stats = FunctionStats::default();

        stats.block_count = self.cfg.blocks.len();

        for block in self.cfg.blocks.values() {
            stats.instruction_count += block.instructions.len();
            stats.phi_count += block.phi_nodes.len();
        }

        // TODO: Compute loop count and max depth

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_creation() {
        let sig = IrFunctionSignature {
            parameters: vec![
                IrParameter {
                    name: "x".to_string(),
                    ty: IrType::I32,
                    reg: IrId::new(0),
                    by_ref: false,
                },
                IrParameter {
                    name: "y".to_string(),
                    ty: IrType::I32,
                    reg: IrId::new(0),
                    by_ref: false,
                },
            ],
            return_type: IrType::I32,
            calling_convention: CallingConvention::Haxe,
            can_throw: false,
            type_params: Vec::new(),
            uses_sret: false,
        };

        let func = IrFunction::new(
            IrFunctionId(1),
            SymbolId::from_raw(100),
            "add".to_string(),
            sig,
        );

        assert_eq!(func.name, "add");
        assert_eq!(func.signature.parameters.len(), 2);
        assert!(func.is_leaf());

        // Parameters should have registers assigned
        assert_ne!(
            func.signature.parameters[0].reg,
            func.signature.parameters[1].reg
        );
    }

    #[test]
    fn test_local_declaration() {
        let sig = IrFunctionSignature {
            parameters: Vec::new(),
            return_type: IrType::Void,
            calling_convention: CallingConvention::Haxe,
            can_throw: false,
            type_params: Vec::new(),
            uses_sret: false,
        };

        let mut func = IrFunction::new(
            IrFunctionId(1),
            SymbolId::from_raw(100),
            "test".to_string(),
            sig,
        );

        let local1 = func.declare_local("tmp1".to_string(), IrType::I32);
        let local2 = func.declare_local("tmp2".to_string(), IrType::Bool);

        assert_ne!(local1, local2);
        assert_eq!(func.locals.len(), 2);
        assert_eq!(func.locals[&local1].name, "tmp1");
        assert_eq!(func.locals[&local2].ty, IrType::Bool);
    }
}
