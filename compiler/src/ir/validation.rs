//! HIR Validation
//!
//! This module provides validation passes for the HIR to ensure correctness
//! before code generation. It checks for type consistency, proper SSA form,
//! control flow integrity, and other invariants.

use super::{
    IrBasicBlock, IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction, IrModule, IrTerminator,
    IrType, IrValue, LifetimeId,
};
use std::collections::{BTreeMap, BTreeSet};

/// Ownership state of a register
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipState {
    /// Register is valid and can be used
    Valid,

    /// Register has been moved and cannot be used
    Moved,

    /// Register is immutably borrowed
    Borrowed,

    /// Register is mutably borrowed (exclusive)
    MutablyBorrowed,

    /// Register has been dropped
    Dropped,

    /// Register is uninitialized
    Uninitialized,
}

/// Borrow information for a register
#[derive(Debug, Clone)]
pub struct BorrowInfo {
    /// The borrowed register
    pub source: IrId,

    /// The borrow (reference) register
    pub borrow: IrId,

    /// Is this a mutable borrow?
    pub is_mutable: bool,

    /// Lifetime of the borrow
    pub lifetime: LifetimeId,

    /// Is the borrow still active?
    pub active: bool,
}

/// HIR validation context
pub struct ValidationContext {
    /// Current module being validated
    module: *const IrModule,

    /// Current function being validated
    current_function: Option<IrFunctionId>,

    /// Errors found during validation
    errors: Vec<ValidationError>,

    /// Type information for each register
    register_types: BTreeMap<IrId, IrType>,

    /// Defined registers
    defined_registers: BTreeSet<IrId>,

    /// Used registers
    used_registers: BTreeSet<IrId>,

    /// Ownership state of each register
    ownership_states: BTreeMap<IrId, OwnershipState>,

    /// Active borrows for each register
    active_borrows: BTreeMap<IrId, Vec<BorrowInfo>>,

    /// Track where each register was moved
    move_locations: BTreeMap<IrId, String>,

    /// Track active lifetimes
    active_lifetimes: BTreeMap<LifetimeId, IrId>,
}

/// Validation error
#[derive(Debug)]
pub struct ValidationError {
    pub kind: ValidationErrorKind,
    pub function: Option<IrFunctionId>,
    pub block: Option<IrBlockId>,
    pub instruction: Option<String>,
}

/// Types of validation errors
#[derive(Debug)]
pub enum ValidationErrorKind {
    /// Register used before definition
    UseBeforeDefine { register: IrId },

    /// Register defined multiple times
    MultipleDefinitions { register: IrId },

    /// Type mismatch
    TypeMismatch {
        expected: IrType,
        found: IrType,
        register: IrId,
    },

    /// Invalid instruction operand
    InvalidOperand { instruction: String, reason: String },

    /// Missing terminator in basic block
    MissingTerminator { block: IrBlockId },

    /// Unreachable code after terminator
    UnreachableCode { block: IrBlockId },

    /// Invalid control flow
    InvalidControlFlow {
        from: IrBlockId,
        to: IrBlockId,
        reason: String,
    },

    /// Phi node inconsistency
    InvalidPhiNode { block: IrBlockId, reason: String },

    /// Function signature mismatch
    SignatureMismatch {
        function: IrFunctionId,
        reason: String,
    },

    /// Invalid SSA form
    InvalidSSA { register: IrId, reason: String },

    /// Memory safety violation: use after move
    UseAfterMove { register: IrId, moved_at: String },

    /// Memory safety violation: use of moved value
    UseOfMovedValue { register: IrId },

    /// Memory safety violation: double move
    DoubleMove { register: IrId, first_move: String },

    /// Borrow checking violation: mutable borrow while immutable borrows exist
    MutableBorrowConflict {
        register: IrId,
        existing_borrows: Vec<IrId>,
    },

    /// Borrow checking violation: multiple mutable borrows
    MultipleMutableBorrows {
        register: IrId,
        existing_borrow: IrId,
    },

    /// Borrow checking violation: use while mutably borrowed
    UseWhileMutablyBorrowed { register: IrId, borrow: IrId },

    /// Lifetime violation: borrow outlives owner
    BorrowOutlivesOwner { borrow: IrId, owner: IrId },

    /// Lifetime violation: dangling reference
    DanglingReference {
        reference: IrId,
        dropped_value: IrId,
    },

    /// Drop of already dropped value
    DoubleDropind { register: IrId },

    /// Drop of borrowed value
    DropWhileBorrowed { register: IrId, borrows: Vec<IrId> },
}

impl ValidationContext {
    /// Create a new validation context
    fn new() -> Self {
        Self {
            module: std::ptr::null(),
            current_function: None,
            errors: Vec::new(),
            register_types: BTreeMap::new(),
            defined_registers: BTreeSet::new(),
            used_registers: BTreeSet::new(),
            ownership_states: BTreeMap::new(),
            active_borrows: BTreeMap::new(),
            move_locations: BTreeMap::new(),
            active_lifetimes: BTreeMap::new(),
        }
    }

    /// Add a validation error
    fn add_error(&mut self, kind: ValidationErrorKind) {
        self.errors.push(ValidationError {
            kind,
            function: self.current_function,
            block: None,
            instruction: None,
        });
    }

    /// Get the module reference
    fn module(&self) -> &IrModule {
        unsafe { &*self.module }
    }

    /// Record register definition with type
    fn define_register(&mut self, reg: IrId, ty: IrType) {
        if self.defined_registers.contains(&reg) {
            self.add_error(ValidationErrorKind::MultipleDefinitions { register: reg });
        }
        self.defined_registers.insert(reg);
        self.register_types.insert(reg, ty);
    }

    /// Record register use and check if defined
    fn use_register(&mut self, reg: IrId) -> Option<&IrType> {
        self.used_registers.insert(reg);

        if !self.defined_registers.contains(&reg) {
            self.add_error(ValidationErrorKind::UseBeforeDefine { register: reg });
            None
        } else {
            self.register_types.get(&reg)
        }
    }

    /// Check type compatibility
    fn check_type_compat(&mut self, expected: &IrType, found: &IrType, reg: IrId) {
        if !types_compatible(expected, found) {
            self.add_error(ValidationErrorKind::TypeMismatch {
                expected: expected.clone(),
                found: found.clone(),
                register: reg,
            });
        }
    }
}

/// Validate an entire module
pub fn validate_module(module: &IrModule) -> Result<(), Vec<ValidationError>> {
    let mut ctx = ValidationContext::new();
    ctx.module = module as *const IrModule;

    // Validate all functions
    for (&func_id, function) in &module.functions {
        ctx.current_function = Some(func_id);
        validate_function(&mut ctx, function);
    }

    // TODO: Validate globals, types, etc.

    if ctx.errors.is_empty() {
        Ok(())
    } else {
        Err(ctx.errors)
    }
}

/// Validate a function
fn validate_function(ctx: &mut ValidationContext, function: &IrFunction) {
    // Clear per-function state
    ctx.register_types.clear();
    ctx.defined_registers.clear();
    ctx.used_registers.clear();

    // Define parameter registers
    for param in &function.signature.parameters {
        ctx.define_register(param.reg, param.ty.clone());
    }

    // Validate control flow graph structure
    validate_cfg_structure(ctx, function);

    // Validate each basic block
    let mut visited = BTreeSet::new();
    validate_block_recursive(ctx, function, function.entry_block(), &mut visited);

    // Check for unused registers (potential dead code)
    for &reg in &ctx.defined_registers {
        if !ctx.used_registers.contains(&reg) && !is_void_type(&ctx.register_types[&reg]) {
            // This is just a warning, not an error
            // Could be reported for optimization
        }
    }

    // Validate return types
    validate_return_types(ctx, function);
}

/// Validate CFG structure
fn validate_cfg_structure(ctx: &mut ValidationContext, function: &IrFunction) {
    // Check entry block exists
    if !function.cfg.blocks.contains_key(&function.entry_block()) {
        ctx.add_error(ValidationErrorKind::InvalidControlFlow {
            from: IrBlockId::entry(),
            to: IrBlockId::entry(),
            reason: "Entry block not found".to_string(),
        });
        return;
    }

    // Check all blocks are reachable from entry
    let reachable = find_reachable_blocks(function);
    for &block_id in function.cfg.blocks.keys() {
        if !reachable.contains(&block_id) && block_id != function.entry_block() {
            // This is a warning - unreachable code
        }
    }

    // Validate predecessor/successor consistency
    for (&block_id, block) in &function.cfg.blocks {
        for &succ in &block.successors() {
            if !function.cfg.blocks.contains_key(&succ) {
                ctx.add_error(ValidationErrorKind::InvalidControlFlow {
                    from: block_id,
                    to: succ,
                    reason: "Successor block not found".to_string(),
                });
            }
        }
    }
}

/// Find all reachable blocks from entry
fn find_reachable_blocks(function: &IrFunction) -> BTreeSet<IrBlockId> {
    let mut reachable = BTreeSet::new();
    let mut worklist = vec![function.entry_block()];

    while let Some(block_id) = worklist.pop() {
        if reachable.insert(block_id) {
            if let Some(block) = function.cfg.get_block(block_id) {
                worklist.extend(block.successors());
            }
        }
    }

    reachable
}

/// Validate a block and its successors recursively
fn validate_block_recursive(
    ctx: &mut ValidationContext,
    function: &IrFunction,
    block_id: IrBlockId,
    visited: &mut BTreeSet<IrBlockId>,
) {
    if !visited.insert(block_id) {
        return; // Already visited
    }

    if let Some(block) = function.cfg.get_block(block_id) {
        validate_block(ctx, block);

        // Recursively validate successors
        for &succ in &block.successors() {
            validate_block_recursive(ctx, function, succ, visited);
        }
    }
}

/// Validate a single basic block
fn validate_block(ctx: &mut ValidationContext, block: &IrBasicBlock) {
    // Validate phi nodes (must come first)
    for phi in &block.phi_nodes {
        validate_phi_node(ctx, block, phi);
    }

    // Validate instructions
    for (i, inst) in block.instructions.iter().enumerate() {
        validate_instruction(ctx, inst);

        // Check for instructions after terminator
        if inst.is_terminator() && i < block.instructions.len() - 1 {
            ctx.add_error(ValidationErrorKind::UnreachableCode { block: block.id });
        }
    }

    // Validate terminator
    if !block.is_terminated() {
        ctx.add_error(ValidationErrorKind::MissingTerminator { block: block.id });
    } else {
        validate_terminator(ctx, &block.terminator);
    }
}

/// Validate a phi node
fn validate_phi_node(ctx: &mut ValidationContext, block: &IrBasicBlock, phi: &super::IrPhiNode) {
    // Check that incoming blocks are predecessors
    for &(pred_block, _) in &phi.incoming {
        if !block.predecessors.contains(&pred_block) {
            ctx.add_error(ValidationErrorKind::InvalidPhiNode {
                block: block.id,
                reason: format!("Incoming block {:?} is not a predecessor", pred_block),
            });
        }
    }

    // Check that all predecessors have entries
    for &pred in &block.predecessors {
        if !phi.incoming.iter().any(|(b, _)| *b == pred) {
            ctx.add_error(ValidationErrorKind::InvalidPhiNode {
                block: block.id,
                reason: format!("Missing entry for predecessor {:?}", pred),
            });
        }
    }

    // Define the phi result
    ctx.define_register(phi.dest, phi.ty.clone());

    // Check all incoming values have compatible types
    for &(_, value) in &phi.incoming {
        if let Some(value_ty) = ctx.use_register(value) {
            let value_ty_clone = value_ty.clone();
            ctx.check_type_compat(&phi.ty, &value_ty_clone, value);
        }
    }
}

/// Validate an instruction
fn validate_instruction(ctx: &mut ValidationContext, inst: &IrInstruction) {
    use IrInstruction::*;

    match inst {
        Const { dest, value } => {
            let ty = value_type(value);
            ctx.define_register(*dest, ty);
        }

        Copy { dest, src } => {
            if let Some(src_ty) = ctx.use_register(*src) {
                let src_ty_clone = src_ty.clone();
                ctx.define_register(*dest, src_ty_clone);
            }
        }

        Load { dest, ptr, ty } => {
            if let Some(ptr_ty) = ctx.use_register(*ptr) {
                // Check that ptr is actually a pointer
                let ptr_ty_clone = ptr_ty.clone();
                match ptr_ty_clone {
                    IrType::Ptr(elem_ty) | IrType::Ref(elem_ty) => {
                        ctx.check_type_compat(&elem_ty, ty, *ptr);
                    }
                    _ => {
                        ctx.add_error(ValidationErrorKind::InvalidOperand {
                            instruction: "Load".to_string(),
                            reason: "Pointer operand is not a pointer type".to_string(),
                        });
                    }
                }
            }
            ctx.define_register(*dest, ty.clone());
        }

        Store { ptr, value, .. } => {
            let ptr_ty = ctx.use_register(*ptr).cloned();
            let val_ty = ctx.use_register(*value).cloned();
            if let (Some(ptr_ty), Some(val_ty)) = (ptr_ty, val_ty) {
                match ptr_ty {
                    IrType::Ptr(elem_ty) | IrType::Ref(elem_ty) => {
                        ctx.check_type_compat(&elem_ty, &val_ty, *value);
                    }
                    _ => {
                        ctx.add_error(ValidationErrorKind::InvalidOperand {
                            instruction: "Store".to_string(),
                            reason: "Pointer operand is not a pointer type".to_string(),
                        });
                    }
                }
            }
        }

        BinOp {
            dest,
            op,
            left,
            right,
        } => {
            let left_ty = ctx.use_register(*left).cloned();
            let right_ty = ctx.use_register(*right).cloned();
            if let (Some(left_ty), Some(right_ty)) = (left_ty, right_ty) {
                // Check operand types are compatible
                if !types_compatible(&left_ty, &right_ty) {
                    ctx.add_error(ValidationErrorKind::TypeMismatch {
                        expected: left_ty.clone(),
                        found: right_ty.clone(),
                        register: *right,
                    });
                }

                // Determine result type
                let result_ty = binary_op_result_type(*op, &left_ty);
                ctx.define_register(*dest, result_ty);
            }
        }

        UnOp { dest, op, operand } => {
            if let Some(operand_ty) = ctx.use_register(*operand) {
                let operand_ty_clone = operand_ty.clone();
                let result_ty = unary_op_result_type(*op, &operand_ty_clone);
                ctx.define_register(*dest, result_ty);
            }
        }

        Cmp {
            dest,
            op: _,
            left,
            right,
        } => {
            let left_ty = ctx.use_register(*left).cloned();
            let right_ty = ctx.use_register(*right).cloned();
            if let (Some(left_ty), Some(right_ty)) = (left_ty, right_ty) {
                if !types_compatible(&left_ty, &right_ty) {
                    ctx.add_error(ValidationErrorKind::TypeMismatch {
                        expected: left_ty.clone(),
                        found: right_ty.clone(),
                        register: *right,
                    });
                }
                ctx.define_register(*dest, IrType::Bool);
            }
        }

        CallDirect {
            dest,
            func_id,
            args,
            arg_ownership: _,
            type_args: _,
            is_tail_call: _,
        } => {
            // For direct calls, we'd need to look up the function signature from the module
            // For now, just validate that arguments are valid registers
            // TODO: Validate ownership modes match function signature
            // TODO: Validate type_args match function's type parameters
            for &arg in args {
                ctx.use_register(arg);
            }

            if let Some(dest_reg) = dest {
                // Define destination register with unknown type (would need module context for real type)
                ctx.define_register(*dest_reg, IrType::Any);
            }
        }

        CallIndirect {
            dest,
            func_ptr,
            args,
            signature,
            arg_ownership: _,
            is_tail_call: _,
        } => {
            // Validate function pointer
            ctx.use_register(*func_ptr);

            // Validate arguments
            for &arg in args {
                ctx.use_register(arg);
            }

            if let Some(dest_reg) = dest {
                // Define destination register with return type from signature
                match signature {
                    IrType::Function { return_type, .. } => {
                        ctx.define_register(*dest_reg, (**return_type).clone());
                    }
                    _ => {
                        ctx.define_register(*dest_reg, IrType::Any);
                    }
                }
            }
        }

        MakeClosure {
            dest,
            func_id: _,
            captured_values,
        } => {
            // Validate all captured values are valid registers
            for &val in captured_values {
                ctx.use_register(val);
            }
            // Closure is represented as a pointer
            ctx.define_register(*dest, IrType::Ptr(Box::new(IrType::Void)));
        }

        ClosureFunc { dest, closure } => {
            // Validate closure is a valid register
            ctx.use_register(*closure);
            // Function pointer is a void*
            ctx.define_register(*dest, IrType::Ptr(Box::new(IrType::Void)));
        }

        ClosureEnv { dest, closure } => {
            // Validate closure is a valid register
            ctx.use_register(*closure);
            // Environment is a void*
            ctx.define_register(*dest, IrType::Ptr(Box::new(IrType::Void)));
        }

        // TODO: Validate remaining instruction types
        _ => {}
    }
}

/// Validate a terminator
fn validate_terminator(ctx: &mut ValidationContext, term: &IrTerminator) {
    use IrTerminator::*;

    match term {
        Branch { .. } => {
            // Nothing to validate
        }

        CondBranch { condition, .. } => {
            if let Some(cond_ty) = ctx.use_register(*condition) {
                let cond_ty_clone = cond_ty.clone();
                if cond_ty_clone != IrType::Bool {
                    ctx.add_error(ValidationErrorKind::TypeMismatch {
                        expected: IrType::Bool,
                        found: cond_ty_clone,
                        register: *condition,
                    });
                }
            }
        }

        Switch { value, .. } => {
            if let Some(val_ty) = ctx.use_register(*value) {
                if !val_ty.is_integer() {
                    ctx.add_error(ValidationErrorKind::InvalidOperand {
                        instruction: "Switch".to_string(),
                        reason: "Switch value must be an integer".to_string(),
                    });
                }
            }
        }

        Return { value } => {
            if let Some(val) = value {
                ctx.use_register(*val);
            }
        }

        _ => {}
    }
}

/// Validate return types
fn validate_return_types(ctx: &mut ValidationContext, function: &IrFunction) {
    let expected_return = &function.signature.return_type;

    // Check all return statements
    for block in function.cfg.blocks.values() {
        if let IrTerminator::Return { value } = &block.terminator {
            match (value, expected_return) {
                (None, IrType::Void) => {} // OK
                (Some(val), ty) if *ty != IrType::Void => {
                    if let Some(val_ty) = ctx.register_types.get(val) {
                        let val_ty_clone = val_ty.clone();
                        ctx.check_type_compat(ty, &val_ty_clone, *val);
                    }
                }
                (None, ty) if *ty != IrType::Void => {
                    ctx.add_error(ValidationErrorKind::SignatureMismatch {
                        function: function.id,
                        reason: "Missing return value".to_string(),
                    });
                }
                (Some(_), IrType::Void) => {
                    ctx.add_error(ValidationErrorKind::SignatureMismatch {
                        function: function.id,
                        reason: "Unexpected return value in void function".to_string(),
                    });
                }
                _ => {}
            }
        }
    }
}

/// Get the type of a value
fn value_type(value: &IrValue) -> IrType {
    match value {
        IrValue::Void => IrType::Void,
        IrValue::Undef => IrType::Any,
        IrValue::Null => IrType::Ptr(Box::new(IrType::Void)),
        IrValue::Bool(_) => IrType::Bool,
        IrValue::I8(_) => IrType::I8,
        IrValue::I16(_) => IrType::I16,
        IrValue::I32(_) => IrType::I32,
        IrValue::I64(_) => IrType::I64,
        IrValue::U8(_) => IrType::U8,
        IrValue::U16(_) => IrType::U16,
        IrValue::U32(_) => IrType::U32,
        IrValue::U64(_) => IrType::U64,
        IrValue::F32(_) => IrType::F32,
        IrValue::F64(_) => IrType::F64,
        IrValue::String(_) => IrType::String,
        IrValue::Array(elems) => {
            if elems.is_empty() {
                IrType::Array(Box::new(IrType::Any), 0)
            } else {
                let elem_ty = value_type(&elems[0]);
                IrType::Array(Box::new(elem_ty), elems.len())
            }
        }
        IrValue::Struct(_) => IrType::Any, // TODO: Proper struct type
        IrValue::Function(_) => IrType::Ptr(Box::new(IrType::Void)), // Function pointer as void*
        IrValue::Closure { .. } => IrType::Ptr(Box::new(IrType::Void)), // Closure as void* (contains func ptr + env)
    }
}

/// Check if two types are compatible
fn types_compatible(a: &IrType, b: &IrType) -> bool {
    match (a, b) {
        (IrType::Any, _) | (_, IrType::Any) => true,
        (a, b) => a == b, // TODO: More sophisticated compatibility
    }
}

/// Check if a type is void
fn is_void_type(ty: &IrType) -> bool {
    matches!(ty, IrType::Void)
}

/// Get result type of binary operation
fn binary_op_result_type(op: super::BinaryOp, operand_ty: &IrType) -> IrType {
    use super::BinaryOp::*;
    match op {
        Add | Sub | Mul | Div | Rem => operand_ty.clone(),
        And | Or | Xor | Shl | Shr | Ushr => operand_ty.clone(),
        FAdd | FSub | FMul | FDiv | FRem => operand_ty.clone(),
    }
}

/// Get result type of unary operation
fn unary_op_result_type(op: super::UnaryOp, operand_ty: &IrType) -> IrType {
    use super::UnaryOp::*;
    match op {
        Neg | Not => operand_ty.clone(),
        FNeg => operand_ty.clone(),
    }
}

// ============================================================================
// MIR Safety Validator
// ============================================================================

use crate::semantic_graph::SemanticGraphs;
use crate::tast::SymbolId;

/// MIR Safety Validator - enforces ownership and memory safety at MIR level
///
/// This validator CONSUMES semantic analysis results instead of re-analyzing.
/// It maps MIR registers to TAST symbols and checks that MIR operations
/// respect the constraints discovered during semantic analysis.
pub struct MirSafetyValidator<'a> {
    /// Semantic analysis results (includes OwnershipGraph, LifetimeAnalyzer, etc.)
    semantic_graphs: &'a SemanticGraphs,

    /// Map from TAST symbols to MIR registers (from module)
    symbol_to_register: &'a BTreeMap<SymbolId, IrId>,

    /// Reverse mapping: MIR register to TAST symbol
    register_to_symbol: &'a BTreeMap<IrId, SymbolId>,

    /// Validation errors
    errors: Vec<ValidationError>,
}

impl<'a> MirSafetyValidator<'a> {
    /// Create a new MIR safety validator
    pub fn new(
        semantic_graphs: &'a SemanticGraphs,
        symbol_to_register: &'a BTreeMap<SymbolId, IrId>,
        register_to_symbol: &'a BTreeMap<IrId, SymbolId>,
    ) -> Self {
        Self {
            semantic_graphs,
            symbol_to_register,
            register_to_symbol,
            errors: Vec::new(),
        }
    }

    /// Validate a MIR module against semantic analysis results
    pub fn validate(
        mir_module: &'a IrModule,
        semantic_graphs: &'a SemanticGraphs,
    ) -> Result<(), Vec<ValidationError>> {
        let mut validator = Self::new(
            semantic_graphs,
            &mir_module.symbol_to_register,
            &mir_module.register_to_symbol,
        );

        // Validate all functions in the module
        for (_func_id, function) in &mir_module.functions {
            validator.validate_function(function);
        }

        if validator.errors.is_empty() {
            Ok(())
        } else {
            Err(validator.errors)
        }
    }

    /// Validate a single function
    fn validate_function(&mut self, function: &IrFunction) {
        // Iterate through all basic blocks
        for (_block_id, block) in &function.cfg.blocks {
            self.validate_block(block);
        }
    }

    /// Validate a single basic block
    fn validate_block(&mut self, block: &IrBasicBlock) {
        // Check each instruction
        for instr in &block.instructions {
            self.validate_instruction(instr);
        }

        // Check terminator
        self.validate_terminator(&block.terminator);
    }

    /// Validate a single instruction against ownership rules
    fn validate_instruction(&mut self, instr: &IrInstruction) {
        use super::IrInstruction::*;

        match instr {
            // Move operations: check if source has been moved
            Copy { dest: _, src } | Move { dest: _, src } => {
                if let Some(&symbol_id) = self.register_to_symbol.get(src) {
                    self.check_not_moved(symbol_id, *src);
                }
            }

            // Store operations: check both source and destination
            Store { ptr, value, .. } => {
                if let Some(&symbol_id) = self.register_to_symbol.get(value) {
                    self.check_not_moved(symbol_id, *value);
                }
                if let Some(&symbol_id) = self.register_to_symbol.get(ptr) {
                    self.check_not_moved(symbol_id, *ptr);
                }
            }

            // Load operations: check pointer validity
            Load {
                dest: _,
                ptr,
                ty: _,
            } => {
                if let Some(&symbol_id) = self.register_to_symbol.get(ptr) {
                    self.check_not_moved(symbol_id, *ptr);
                }
            }

            // Binary/Unary operations: check operands
            BinOp {
                dest: _,
                op: _,
                left,
                right,
            } => {
                if let Some(&symbol_id) = self.register_to_symbol.get(left) {
                    self.check_not_moved(symbol_id, *left);
                }
                if let Some(&symbol_id) = self.register_to_symbol.get(right) {
                    self.check_not_moved(symbol_id, *right);
                }
            }

            UnOp {
                dest: _,
                op: _,
                operand,
            } => {
                if let Some(&symbol_id) = self.register_to_symbol.get(operand) {
                    self.check_not_moved(symbol_id, *operand);
                }
            }

            // Function calls: check all arguments
            CallDirect {
                dest: _,
                func_id: _,
                args,
                arg_ownership: _,
                type_args: _,
                is_tail_call: _,
            }
            | CallIndirect {
                dest: _,
                func_ptr: _,
                args,
                signature: _,
                arg_ownership: _,
                is_tail_call: _,
            } => {
                for arg in args {
                    if let Some(&symbol_id) = self.register_to_symbol.get(arg) {
                        self.check_not_moved(symbol_id, *arg);
                    }
                }
            }

            // Other instructions don't directly affect ownership
            _ => {}
        }
    }

    /// Validate a terminator instruction
    fn validate_terminator(&mut self, terminator: &IrTerminator) {
        use super::IrTerminator::*;

        match terminator {
            Return { value: Some(val) } => {
                if let Some(&symbol_id) = self.register_to_symbol.get(val) {
                    self.check_not_moved(symbol_id, *val);
                }
            }

            CondBranch { condition, .. } => {
                if let Some(&symbol_id) = self.register_to_symbol.get(condition) {
                    self.check_not_moved(symbol_id, *condition);
                }
            }

            Switch { value, .. } => {
                if let Some(&symbol_id) = self.register_to_symbol.get(value) {
                    self.check_not_moved(symbol_id, *value);
                }
            }

            _ => {}
        }
    }

    /// Check if a symbol has been moved (use-after-move detection)
    fn check_not_moved(&mut self, symbol_id: SymbolId, register: IrId) {
        // Query the ownership graph from semantic analysis
        if let Some(ownership_node) = self
            .semantic_graphs
            .ownership_graph
            .variables
            .get(&symbol_id)
        {
            use crate::semantic_graph::OwnershipKind;

            // If the ownership kind indicates the value was moved, report error
            if matches!(ownership_node.ownership_kind, OwnershipKind::Moved) {
                self.errors.push(ValidationError {
                    kind: ValidationErrorKind::UseOfMovedValue { register },
                    function: None,
                    block: None,
                    instruction: None,
                });
            }
        }
    }

    /// Get symbol ID for a register
    fn _register_to_symbol(&self, register: IrId) -> Option<SymbolId> {
        self.register_to_symbol.get(&register).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::*;
    use crate::tast::SymbolId;

    #[test]
    fn test_validate_simple_function() {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());

        let sig = FunctionSignatureBuilder::new()
            .param("x".to_string(), IrType::I32)
            .returns(IrType::I32)
            .build();

        builder.start_function(SymbolId::from_raw(1), "identity".to_string(), sig);

        let x = builder
            .current_function()
            .unwrap()
            .get_param_reg(0)
            .unwrap();
        builder.build_return(Some(x));

        builder.finish_function();

        // Validate the module
        let result = validate_module(&builder.module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_type_mismatch() {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());

        let sig = FunctionSignatureBuilder::new().returns(IrType::I32).build();

        builder.start_function(SymbolId::from_raw(1), "bad_return".to_string(), sig);

        // Return a boolean instead of i32
        let bool_val = builder.build_bool(true).unwrap();
        builder.build_return(Some(bool_val));

        builder.finish_function();

        // Validate the module
        let result = validate_module(&builder.module);
        assert!(result.is_err());

        if let Err(errors) = result {
            assert!(!errors.is_empty());
            // Should have type mismatch error
        }
    }
}
