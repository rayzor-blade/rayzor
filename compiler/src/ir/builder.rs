//! HIR Builder
//!
//! This module provides a builder interface for constructing HIR in a convenient way.
//! The builder maintains context and provides helper methods for common patterns.

use tracing::debug;

use super::{
    AllocationHint, BinaryOp, CallingConvention, CompareOp, IrBasicBlock, IrBlockId, IrFunction,
    IrFunctionId, IrFunctionSignature, IrId, IrInstruction, IrLocal, IrModule, IrParameter,
    IrPhiNode, IrSourceLocation, IrTerminator, IrType, IrTypeParam, IrValue, UnaryOp,
};
use crate::tast::SymbolId;
use std::collections::HashMap;

/// HIR builder for constructing IR modules
pub struct IrBuilder {
    /// The module being built
    pub module: IrModule,

    /// Current function being built
    pub(crate) current_function: Option<IrFunctionId>,

    /// Current basic block being built
    pub(crate) current_block: Option<IrBlockId>,

    /// Source location context
    current_source_location: IrSourceLocation,
}

impl IrBuilder {
    /// Create a new IR builder
    pub fn new(module_name: String, source_file: String) -> Self {
        Self {
            module: IrModule::new(module_name, source_file),
            current_function: None,
            current_block: None,
            current_source_location: IrSourceLocation::unknown(),
        }
    }

    /// Set the current source location for debugging
    pub fn set_source_location(&mut self, loc: IrSourceLocation) {
        self.current_source_location = loc;
    }

    // === Module Building ===

    /// Start building a new function
    pub fn start_function(
        &mut self,
        symbol_id: SymbolId,
        name: String,
        signature: IrFunctionSignature,
    ) -> IrFunctionId {
        let id = self.module.alloc_function_id();
        let function = IrFunction::new(id, symbol_id, name, signature);
        self.current_function = Some(id);
        self.current_block = Some(function.entry_block());
        self.module.add_function(function);
        id
    }

    /// Finish building the current function
    pub fn finish_function(&mut self) {
        self.current_function = None;
        self.current_block = None;
    }

    /// Get the current function
    pub fn current_function(&self) -> Option<&IrFunction> {
        self.current_function
            .and_then(|id| self.module.functions.get(&id))
    }

    /// Get the current function mutably
    pub fn current_function_mut(&mut self) -> Option<&mut IrFunction> {
        self.current_function
            .and_then(move |id| self.module.functions.get_mut(&id))
    }

    // === Block Building ===

    /// Create a new basic block in the current function
    pub fn create_block(&mut self) -> Option<IrBlockId> {
        self.current_function_mut().map(|f| f.cfg.create_block())
    }

    /// Create a new basic block with a label
    pub fn create_block_with_label(&mut self, label: String) -> Option<IrBlockId> {
        let block_id = self.create_block()?;
        self.current_function_mut()
            .and_then(|f| f.cfg.get_block_mut(block_id))
            .map(|b| b.label = Some(label));
        Some(block_id)
    }

    /// Switch to building in a different block
    pub fn switch_to_block(&mut self, block: IrBlockId) {
        self.current_block = Some(block);
    }

    /// Get the current block
    pub fn current_block(&self) -> Option<IrBlockId> {
        self.current_block
    }

    // === Register Management ===

    /// Allocate a new register in the current function
    pub fn alloc_reg(&mut self) -> Option<IrId> {
        self.current_function_mut().map(|f| f.alloc_reg())
    }

    /// Get the type of a register
    pub fn get_register_type(&self, reg: IrId) -> Option<IrType> {
        let func_id = self.current_function?;
        let func = self.module.functions.get(&func_id)?;
        func.register_types.get(&reg).cloned()
    }

    /// Set the type of a register
    pub fn set_register_type(&mut self, reg: IrId, ty: IrType) {
        if let Some(func) = self.current_function_mut() {
            func.register_types.insert(reg, ty);
        }
    }

    /// Declare a local variable
    pub fn declare_local(&mut self, name: String, ty: IrType) -> Option<IrId> {
        self.current_function_mut()
            .map(|f| f.declare_local(name, ty))
    }

    // === Instruction Building ===

    /// Add an instruction to the current block
    fn add_instruction(&mut self, inst: IrInstruction) -> Option<()> {
        let block_id = self.current_block?;
        self.current_function_mut()
            .and_then(|f| f.cfg.get_block_mut(block_id))
            .map(|b| b.add_instruction(inst))
    }

    /// Build a constant instruction
    pub fn build_const(&mut self, value: IrValue) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        // Infer type from value and store in register types
        let ty = match &value {
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
            IrValue::Bool(_) => IrType::Bool,
            IrValue::String(_) => IrType::String,
            _ => IrType::Any,
        };
        self.set_register_type(dest, ty);
        self.add_instruction(IrInstruction::Const { dest, value })?;
        Some(dest)
    }

    /// Build a function pointer constant
    pub fn build_function_ptr(&mut self, func_id: IrFunctionId) -> Option<IrId> {
        let value = IrValue::Function(func_id);
        self.build_const(value)
    }

    /// Build a copy instruction
    pub fn build_copy(&mut self, src: IrId) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        // Propagate type from source for LLVM backend
        if let Some(ty) = self.get_register_type(src) {
            self.set_register_type(dest, ty);
        }
        self.add_instruction(IrInstruction::Copy { dest, src })?;
        Some(dest)
    }

    /// Build a load instruction
    pub fn build_load(&mut self, ptr: IrId, ty: IrType) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        // Track the type of the loaded value
        self.set_register_type(dest, ty.clone());
        self.add_instruction(IrInstruction::Load { dest, ptr, ty })?;
        Some(dest)
    }

    /// Build a store instruction
    pub fn build_store(&mut self, ptr: IrId, value: IrId) -> Option<()> {
        self.add_instruction(IrInstruction::Store { ptr, value })
    }

    /// Build a load from global variable
    pub fn build_load_global(&mut self, global_id: super::IrGlobalId, ty: IrType) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.set_register_type(dest, ty.clone());
        self.add_instruction(IrInstruction::LoadGlobal {
            dest,
            global_id,
            ty,
        })?;
        Some(dest)
    }

    /// Build a store to global variable
    pub fn build_store_global(&mut self, global_id: super::IrGlobalId, value: IrId) -> Option<()> {
        self.add_instruction(IrInstruction::StoreGlobal { global_id, value })
    }

    /// Build a binary operation
    pub fn build_binop(&mut self, op: BinaryOp, left: IrId, right: IrId) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        // Infer result type from left operand (or right if left is unknown)
        // This is critical for LLVM backend to know whether to use int or float ops
        if let Some(ty) = self
            .get_register_type(left)
            .or_else(|| self.get_register_type(right))
        {
            self.set_register_type(dest, ty);
        }
        self.add_instruction(IrInstruction::BinOp {
            dest,
            op,
            left,
            right,
        })?;
        Some(dest)
    }

    /// Build a unary operation
    pub fn build_unop(&mut self, op: UnaryOp, operand: IrId) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        // Infer result type from operand for LLVM backend
        if let Some(ty) = self.get_register_type(operand) {
            self.set_register_type(dest, ty);
        }
        self.add_instruction(IrInstruction::UnOp { dest, op, operand })?;
        Some(dest)
    }

    /// Build a comparison operation
    pub fn build_cmp(&mut self, op: CompareOp, left: IrId, right: IrId) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        // Comparisons always return Bool
        self.set_register_type(dest, IrType::Bool);
        self.add_instruction(IrInstruction::Cmp {
            dest,
            op,
            left,
            right,
        })?;
        Some(dest)
    }

    /// Build a SIMD vector binary operation (element-wise)
    pub fn build_vector_binop(
        &mut self,
        op: BinaryOp,
        left: IrId,
        right: IrId,
        vec_ty: IrType,
    ) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.set_register_type(dest, vec_ty.clone());
        self.add_instruction(IrInstruction::VectorBinOp {
            dest,
            op,
            left,
            right,
            vec_ty,
        })?;
        Some(dest)
    }

    /// Build a function call
    /// Build a direct function call (callee known at compile time)
    pub fn build_call_direct(
        &mut self,
        func_id: IrFunctionId,
        args: Vec<IrId>,
        ty: IrType,
    ) -> Option<IrId> {
        // IMPORTANT: Always use the function's actual signature if available.
        // The caller might pass an incorrect type (e.g., Any instead of Void) due to
        // type inference issues in earlier stages. Using the actual signature ensures
        // that void functions don't get destination registers allocated.
        let actual_return_type = if let Some(func) = self.module.functions.get(&func_id) {
            func.signature.return_type.clone()
        } else if let Some(extern_func) = self.module.extern_functions.get(&func_id) {
            // Check extern functions too (for runtime functions)
            extern_func.signature.return_type.clone()
        } else {
            // Function not in module yet (might be a forward ref being called from a lambda).
            // Fall back to the provided type, but this is a known limitation.
            // TODO: Pass the HirToMirContext to IrBuilder so we can lookup stdlib signatures here.
            ty.clone()
        };

        // Coerce argument types to match the function's parameter types.
        // This handles implicit int→float conversions (e.g., passing integer 0
        // to a Float parameter) which Haxe allows but the backend requires
        // matching types. Without this, inlining propagates the wrong type
        // into store instructions (e.g., storing i32 into an f64 field).
        let param_types: Vec<IrType> = self
            .module
            .functions
            .get(&func_id)
            .map(|f| {
                f.signature
                    .parameters
                    .iter()
                    .map(|p| p.ty.clone())
                    .collect()
            })
            .or_else(|| {
                self.module.extern_functions.get(&func_id).map(|f| {
                    f.signature
                        .parameters
                        .iter()
                        .map(|p| p.ty.clone())
                        .collect()
                })
            })
            .unwrap_or_default();

        let mut final_args = args;
        for (i, arg_id) in final_args.iter_mut().enumerate() {
            if let Some(param_ty) = param_types.get(i) {
                if let Some(arg_ty) = self.get_register_type(*arg_id) {
                    let needs_cast = matches!(
                        (&arg_ty, param_ty),
                        (IrType::I32 | IrType::I64, IrType::F64 | IrType::F32)
                            | (IrType::F64 | IrType::F32, IrType::I32 | IrType::I64)
                            | (IrType::I64, IrType::Ptr(_))
                            | (IrType::Ptr(_), IrType::I64)
                    );
                    if needs_cast {
                        // F64↔I64: use bitcast (bit-preserving) for type erasure boundaries
                        // Other cases: use cast (value conversion)
                        let use_bitcast = matches!(
                            (&arg_ty, param_ty),
                            (IrType::F64, IrType::I64) | (IrType::I64, IrType::F64)
                        );
                        let cast_id = if use_bitcast {
                            self.build_bitcast(*arg_id, param_ty.clone())
                        } else {
                            self.build_cast(*arg_id, arg_ty, param_ty.clone())
                        };
                        if let Some(cast_id) = cast_id {
                            *arg_id = cast_id;
                        }
                    }
                }
            }
        }

        // Only allocate a destination register if the function returns a value
        let dest = if actual_return_type == IrType::Void {
            None
        } else {
            Some(self.alloc_reg()?)
        };

        // Default to Move ownership for all arguments (will be refined by HIR lowering)
        let arg_ownership = final_args
            .iter()
            .map(|_| crate::ir::instructions::OwnershipMode::Move)
            .collect();
        self.add_instruction(IrInstruction::CallDirect {
            dest,
            func_id,
            args: final_args,
            arg_ownership,
            type_args: Vec::new(),
            is_tail_call: false,
        })?;

        // Register the return type for the result register
        if let Some(dest_reg) = dest {
            // Always register the actual return type (important for type-dependent optimizations)
            self.set_register_type(dest_reg, actual_return_type.clone());

            // Check if function's actual return type matches expected type
            // If not, insert a Cast instruction (but NOT if it would lose type precision)
            if actual_return_type != ty {
                // Don't cast from pointer to scalar - that's a bug in type inference for generics
                // When the actual return type is a pointer (class instance) but expected type is
                // a scalar like I32, it means generic type resolution failed. Trust the actual type.
                let actual_is_ptr = matches!(actual_return_type, IrType::Ptr(_));
                let actual_is_scalar = matches!(
                    actual_return_type,
                    IrType::I32 | IrType::I64 | IrType::Bool | IrType::F32 | IrType::F64
                );
                let expected_is_scalar = matches!(
                    ty,
                    IrType::I32 | IrType::I64 | IrType::Bool | IrType::F32 | IrType::F64
                );
                let expected_is_ptr = matches!(ty, IrType::Ptr(_));

                let actual_is_vector = actual_return_type.is_vector();

                if actual_is_vector {
                    // Vector types (e.g., F32X4 from SIMD4f) should never be cast.
                    // The function returns the correct vector type; the caller's expected
                    // type may be Ptr(Void) due to abstract type resolution.
                    debug!("DEBUG: CallDirect type mismatch - function returns {:?}, expected {:?}, but NOT inserting cast (vector type)",
                              actual_return_type, ty);
                } else if actual_is_ptr && expected_is_scalar {
                    debug!("DEBUG: CallDirect type mismatch - function returns {:?}, expected {:?}, but NOT inserting cast (pointer->scalar would lose data)",
                              actual_return_type, ty);
                    // Type already registered above - trust actual type
                } else if actual_is_scalar && expected_is_ptr {
                    // Don't cast from scalar to pointer - the function returns a concrete type
                    // (e.g., F64 from Sys.time()) but the HIR expects Dynamic (Ptr(Void)).
                    // Trust the actual return type to preserve type information.
                    debug!("DEBUG: CallDirect type mismatch - function returns {:?}, expected {:?}, but NOT inserting cast (scalar->pointer would lose type info)",
                              actual_return_type, ty);
                    // Type already registered above - trust actual type
                } else {
                    debug!("DEBUG: CallDirect type mismatch - function returns {:?}, expected {:?}, inserting cast",
                              actual_return_type, ty);
                    // I64↔F64: use bitcast (bit-preserving) for type erasure boundaries
                    let use_bitcast = matches!(
                        (&actual_return_type, &ty),
                        (IrType::I64, IrType::F64) | (IrType::F64, IrType::I64)
                    );
                    if use_bitcast {
                        return self.build_bitcast(dest_reg, ty);
                    }
                    return self.build_cast(dest_reg, actual_return_type.clone(), ty);
                }
            }
        }

        dest
    }

    /// Build a direct function call with generic type arguments
    ///
    /// This is used for calls to generic functions where the type arguments
    /// are known at compile time. The monomorphization pass will use these
    /// type_args to generate specialized function instantiations.
    pub fn build_call_direct_with_type_args(
        &mut self,
        func_id: IrFunctionId,
        args: Vec<IrId>,
        ty: IrType,
        type_args: Vec<IrType>,
    ) -> Option<IrId> {
        // Same logic as build_call_direct but with type_args
        let actual_return_type = if let Some(func) = self.module.functions.get(&func_id) {
            func.signature.return_type.clone()
        } else if let Some(extern_func) = self.module.extern_functions.get(&func_id) {
            extern_func.signature.return_type.clone()
        } else {
            ty.clone()
        };

        let dest = if actual_return_type == IrType::Void {
            None
        } else {
            Some(self.alloc_reg()?)
        };

        let arg_ownership = args
            .iter()
            .map(|_| crate::ir::instructions::OwnershipMode::Move)
            .collect();
        self.add_instruction(IrInstruction::CallDirect {
            dest,
            func_id,
            args,
            arg_ownership,
            type_args,
            is_tail_call: false,
        })?;

        if let Some(dest_reg) = dest {
            self.set_register_type(dest_reg, actual_return_type.clone());

            if actual_return_type != ty {
                let actual_is_ptr = matches!(actual_return_type, IrType::Ptr(_));
                let expected_is_scalar = matches!(
                    ty,
                    IrType::I32 | IrType::I64 | IrType::Bool | IrType::F32 | IrType::F64
                );

                if actual_is_ptr && expected_is_scalar {
                    debug!("DEBUG: CallDirect (generic) type mismatch - function returns {:?}, expected {:?}, NOT inserting cast",
                              actual_return_type, ty);
                } else {
                    debug!("DEBUG: CallDirect (generic) type mismatch - function returns {:?}, expected {:?}, inserting cast",
                              actual_return_type, ty);
                    return self.build_cast(dest_reg, actual_return_type.clone(), ty);
                }
            }
        }

        dest
    }

    /// Build an indirect function call (callee computed at runtime)
    pub fn build_call_indirect(
        &mut self,
        func_ptr: IrId,
        args: Vec<IrId>,
        signature: IrType,
    ) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        // Default to Move ownership for all arguments
        let arg_ownership = args
            .iter()
            .map(|_| crate::ir::instructions::OwnershipMode::Move)
            .collect();
        self.add_instruction(IrInstruction::CallIndirect {
            dest: Some(dest),
            func_ptr,
            args,
            signature,
            arg_ownership,
            is_tail_call: false,
        })?;
        Some(dest)
    }

    /// Build a cast instruction
    pub fn build_cast(&mut self, src: IrId, from_ty: IrType, to_ty: IrType) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::Cast {
            dest,
            src,
            from_ty: from_ty.clone(),
            to_ty: to_ty.clone(),
        })?;
        // Track the result type
        self.set_register_type(dest, to_ty);
        Some(dest)
    }

    /// Build a bitcast instruction (reinterpret bits without conversion)
    /// Used for type-punning like f64 -> u64 (same bits, different interpretation)
    pub fn build_bitcast(&mut self, src: IrId, to_ty: IrType) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::BitCast {
            dest,
            src,
            ty: to_ty.clone(),
        })?;
        // Track the result type
        self.set_register_type(dest, to_ty);
        Some(dest)
    }

    /// Build an alloc instruction (stack allocation)
    pub fn build_alloc(&mut self, ty: IrType, count: Option<IrId>) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::Alloc { dest, ty, count })?;
        Some(dest)
    }

    /// Get a function by name from the module
    pub fn get_function_by_name(&self, name: &str) -> Option<IrFunctionId> {
        self.module
            .functions
            .iter()
            .find(|(_, f)| f.name == name)
            .map(|(id, _)| *id)
    }

    /// Build a heap allocation by calling malloc
    /// This is used for class instances that may escape the current function
    pub fn build_heap_alloc(&mut self, size: u64) -> Option<IrId> {
        // Get malloc function ID
        let malloc_id = self.get_function_by_name("malloc")?;

        // Create size constant
        let size_reg = self.build_const(IrValue::U64(size))?;

        // Call malloc
        let ptr_u8_ty = IrType::Ptr(Box::new(IrType::U8));
        self.build_call_direct(malloc_id, vec![size_reg], ptr_u8_ty)
    }

    /// Build a free instruction (marks pointer for deallocation)
    /// This emits an IrInstruction::Free which backends translate to actual deallocation
    pub fn build_free(&mut self, ptr: IrId) -> Option<()> {
        self.add_instruction(IrInstruction::Free { ptr })
    }

    /// Build a heap free by calling the free function
    /// This is used for explicit deallocation of heap-allocated objects (Rust-style drop)
    pub fn build_heap_free(&mut self, ptr: IrId) -> Option<()> {
        // Get free function ID
        let free_id = self.get_function_by_name("free")?;

        // Call free(ptr) - returns void
        self.build_call_direct(free_id, vec![ptr], IrType::Void)?;
        Some(())
    }

    /// Build a GEP (get element pointer) instruction
    pub fn build_gep(&mut self, ptr: IrId, indices: Vec<IrId>, ty: IrType) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::GetElementPtr {
            dest,
            ptr,
            indices,
            ty,
        })?;
        Some(dest)
    }

    /// Build a pointer add (byte-offset pointer arithmetic)
    pub fn build_ptr_add(&mut self, ptr: IrId, offset: IrId, ty: IrType) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::PtrAdd {
            dest,
            ptr,
            offset,
            ty,
        })?;
        Some(dest)
    }

    /// Build a function reference (get a function pointer)
    pub fn build_function_ref(&mut self, func_id: super::IrFunctionId) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.set_register_type(dest, IrType::I64);
        self.add_instruction(IrInstruction::FunctionRef { dest, func_id })?;
        Some(dest)
    }

    /// Build a select (ternary) instruction
    pub fn build_select(
        &mut self,
        condition: IrId,
        true_val: IrId,
        false_val: IrId,
    ) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::Select {
            dest,
            condition,
            true_val,
            false_val,
        })?;
        Some(dest)
    }

    /// Build an extract value instruction for accessing aggregate elements
    pub fn build_extract_value(&mut self, aggregate: IrId, indices: Vec<u32>) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::ExtractValue {
            dest,
            aggregate,
            indices,
        })?;
        Some(dest)
    }

    /// Build a closure creation instruction
    pub fn build_make_closure(
        &mut self,
        func_id: IrFunctionId,
        captured_values: Vec<IrId>,
    ) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::MakeClosure {
            dest,
            func_id,
            captured_values,
        })?;
        Some(dest)
    }

    /// Build an instruction to extract the function pointer from a closure
    pub fn build_closure_func(&mut self, closure: IrId) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::ClosureFunc { dest, closure })?;
        Some(dest)
    }

    /// Build an instruction to extract the environment from a closure
    pub fn build_closure_env(&mut self, closure: IrId) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::ClosureEnv { dest, closure })?;
        Some(dest)
    }

    // === Exception Handling ===

    /// Build a throw instruction
    pub fn build_throw(&mut self, exception: IrId) -> Option<()> {
        self.add_instruction(IrInstruction::Throw { exception })?;
        Some(())
    }

    /// Build a landing pad instruction for exception handling
    pub fn build_landing_pad(
        &mut self,
        ty: IrType,
        clauses: Vec<super::LandingPadClause>,
    ) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        self.add_instruction(IrInstruction::LandingPad { dest, ty, clauses })?;
        Some(dest)
    }

    // === Terminator Building ===

    /// Set the terminator for the current block
    fn set_terminator(&mut self, term: IrTerminator) -> Option<()> {
        let block_id = self.current_block?;
        // eprintln!("DEBUG set_terminator: block={:?}, term={:?}", block_id, term);
        let func = self.current_function_mut()?;
        // eprintln!("DEBUG set_terminator: function={}", func.name);

        // First, set the terminator
        let block = func.cfg.get_block_mut(block_id)?;
        block.set_terminator(term.clone());
        // eprintln!("DEBUG set_terminator: terminator set on block successfully");

        // Then, update predecessor information based on the terminator
        match &term {
            IrTerminator::Branch { target } => {
                func.cfg.connect_blocks(block_id, *target);
            }
            IrTerminator::CondBranch {
                true_target,
                false_target,
                ..
            } => {
                func.cfg.connect_blocks(block_id, *true_target);
                func.cfg.connect_blocks(block_id, *false_target);
            }
            IrTerminator::Switch { cases, default, .. } => {
                for (_, target) in cases {
                    func.cfg.connect_blocks(block_id, *target);
                }
                func.cfg.connect_blocks(block_id, *default);
            }
            _ => {}
        }

        Some(())
    }

    /// Build an unconditional branch
    pub fn build_branch(&mut self, target: IrBlockId) -> Option<()> {
        self.set_terminator(IrTerminator::Branch { target })
    }

    /// Build a conditional branch
    pub fn build_cond_branch(
        &mut self,
        condition: IrId,
        true_target: IrBlockId,
        false_target: IrBlockId,
    ) -> Option<()> {
        self.set_terminator(IrTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        })
    }

    /// Build a switch statement
    pub fn build_switch(
        &mut self,
        value: IrId,
        cases: Vec<(i64, IrBlockId)>,
        default: IrBlockId,
    ) -> Option<()> {
        self.set_terminator(IrTerminator::Switch {
            value,
            cases,
            default,
        })
    }

    /// Build a return instruction
    pub fn build_return(&mut self, value: Option<IrId>) -> Option<()> {
        // eprintln!("DEBUG IrBuilder::build_return called with value={:?}", value);
        // eprintln!("DEBUG   Current function: {:?}", self.current_function().map(|f| &f.name));
        // eprintln!("DEBUG   Current block: {:?}", self.current_block);
        let result = self.set_terminator(IrTerminator::Return { value });
        // eprintln!("DEBUG   set_terminator returned: {:?}", result);
        result
    }

    /// Build an unreachable terminator
    pub fn build_unreachable(&mut self) -> Option<()> {
        self.set_terminator(IrTerminator::Unreachable)
    }

    // === Phi Node Building ===

    /// Add a phi node to a block
    pub fn build_phi(&mut self, block: IrBlockId, ty: IrType) -> Option<IrId> {
        let dest = self.alloc_reg()?;
        let phi = IrPhiNode {
            dest,
            incoming: Vec::new(),
            ty,
        };

        self.current_function_mut()
            .and_then(|f| f.cfg.get_block_mut(block))
            .map(|b| b.add_phi(phi))?;

        Some(dest)
    }

    /// Add an incoming value to a phi node
    pub fn add_phi_incoming(
        &mut self,
        block: IrBlockId,
        phi_dest: IrId,
        from_block: IrBlockId,
        value: IrId,
    ) -> Option<()> {
        self.current_function_mut()
            .and_then(|f| f.cfg.get_block_mut(block))
            .and_then(|b| b.phi_nodes.iter_mut().find(|p| p.dest == phi_dest))
            .map(|phi| phi.incoming.push((from_block, value)))
    }

    // === Convenience Methods ===

    /// Build an integer constant
    pub fn build_int(&mut self, value: i64, ty: IrType) -> Option<IrId> {
        let ir_value = match ty {
            IrType::I8 => IrValue::I8(value as i8),
            IrType::I16 => IrValue::I16(value as i16),
            IrType::I32 => IrValue::I32(value as i32),
            IrType::I64 => IrValue::I64(value),
            IrType::U8 => IrValue::U8(value as u8),
            IrType::U16 => IrValue::U16(value as u16),
            IrType::U32 => IrValue::U32(value as u32),
            IrType::U64 => IrValue::U64(value as u64),
            _ => return None,
        };
        self.build_const(ir_value)
    }

    /// Build a boolean constant
    pub fn build_bool(&mut self, value: bool) -> Option<IrId> {
        self.build_const(IrValue::Bool(value))
    }

    /// Build a string constant
    pub fn build_string(&mut self, value: String) -> Option<IrId> {
        // Add to string pool
        let _string_id = self.module.string_pool.add(value.clone());
        self.build_const(IrValue::String(value))
    }

    /// Build a null pointer constant
    pub fn build_null(&mut self) -> Option<IrId> {
        self.build_const(IrValue::Null)
    }

    /// Build addition
    pub fn build_add(&mut self, left: IrId, right: IrId, is_float: bool) -> Option<IrId> {
        let op = if is_float {
            BinaryOp::FAdd
        } else {
            BinaryOp::Add
        };
        self.build_binop(op, left, right)
    }

    /// Build subtraction
    pub fn build_sub(&mut self, left: IrId, right: IrId, is_float: bool) -> Option<IrId> {
        let op = if is_float {
            BinaryOp::FSub
        } else {
            BinaryOp::Sub
        };
        self.build_binop(op, left, right)
    }

    /// Build multiplication
    pub fn build_mul(&mut self, left: IrId, right: IrId, is_float: bool) -> Option<IrId> {
        let op = if is_float {
            BinaryOp::FMul
        } else {
            BinaryOp::Mul
        };
        self.build_binop(op, left, right)
    }

    /// Build division
    pub fn build_div(&mut self, left: IrId, right: IrId, is_float: bool) -> Option<IrId> {
        let op = if is_float {
            BinaryOp::FDiv
        } else {
            BinaryOp::Div
        };
        self.build_binop(op, left, right)
    }

    // === Type Tracking Helpers (for lambda generation) ===

    /// Register a local variable with type tracking
    /// This should be called by instruction builders that create registers
    pub(crate) fn register_local(&mut self, reg: IrId, ty: IrType) -> Option<()> {
        let func = self.current_function_mut()?;

        func.locals.insert(
            reg,
            IrLocal {
                name: format!("r{}", reg.0),
                ty,
                mutable: false,
                source_location: IrSourceLocation::unknown(),
                allocation: AllocationHint::Register,
            },
        );

        Some(())
    }

    /// Infer the result type of a binary operation
    pub(crate) fn infer_binop_type(&self, op: BinaryOp, left: IrId, right: IrId) -> Option<IrType> {
        let func = self.current_function()?;
        let left_ty = &func.locals.get(&left)?.ty;
        let right_ty = &func.locals.get(&right)?.ty;

        // Type inference rules
        Some(match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                // Arithmetic: result type matches operands (prefer I64 if mixed)
                if left_ty == right_ty {
                    left_ty.clone()
                } else {
                    IrType::I64
                }
            }
            BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => {
                // Bitwise: result type matches operands
                if left_ty == right_ty {
                    left_ty.clone()
                } else {
                    IrType::I64
                }
            }
            BinaryOp::FAdd | BinaryOp::FSub | BinaryOp::FMul | BinaryOp::FDiv | BinaryOp::FRem => {
                // Float arithmetic: prefer F64
                IrType::F64
            }
            _ => IrType::I64, // Default for other ops
        })
    }
}

/// Function builder helper for building function signatures
pub struct FunctionSignatureBuilder {
    parameters: Vec<IrParameter>,
    return_type: IrType,
    calling_convention: CallingConvention,
    can_throw: bool,
    type_params: Vec<IrTypeParam>,
}

impl FunctionSignatureBuilder {
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            return_type: IrType::Void,
            calling_convention: CallingConvention::Haxe,
            can_throw: false,
            type_params: Vec::new(),
        }
    }

    /// Add a type parameter to the function signature
    pub fn type_param(mut self, name: String) -> Self {
        self.type_params.push(IrTypeParam {
            name,
            constraints: Vec::new(),
        });
        self
    }

    /// Add a type parameter with constraints
    pub fn type_param_with_constraints(mut self, name: String, constraints: Vec<String>) -> Self {
        self.type_params.push(IrTypeParam { name, constraints });
        self
    }

    pub fn param(mut self, name: String, ty: IrType) -> Self {
        self.parameters.push(IrParameter {
            name,
            ty,
            reg: IrId::new(0), // Will be assigned later
            by_ref: false,
        });
        self
    }

    pub fn param_by_ref(mut self, name: String, ty: IrType) -> Self {
        self.parameters.push(IrParameter {
            name,
            ty,
            reg: IrId::new(0), // Will be assigned later
            by_ref: true,
        });
        self
    }

    pub fn returns(mut self, ty: IrType) -> Self {
        self.return_type = ty;
        self
    }

    pub fn calling_convention(mut self, cc: CallingConvention) -> Self {
        self.calling_convention = cc;
        self
    }

    pub fn can_throw(mut self, throws: bool) -> Self {
        self.can_throw = throws;
        self
    }

    pub fn build(self) -> IrFunctionSignature {
        IrFunctionSignature {
            parameters: self.parameters,
            return_type: self.return_type,
            calling_convention: self.calling_convention,
            can_throw: self.can_throw,
            type_params: self.type_params,
            uses_sret: false, // Generic builder - caller can set manually if needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_function_building() {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());

        // Build a simple add function
        let sig = FunctionSignatureBuilder::new()
            .param("a".to_string(), IrType::I32)
            .param("b".to_string(), IrType::I32)
            .returns(IrType::I32)
            .build();

        let func_id = builder.start_function(SymbolId::from_raw(1), "add".to_string(), sig);

        // Get parameter registers
        let a = builder
            .current_function()
            .unwrap()
            .get_param_reg(0)
            .unwrap();
        let b = builder
            .current_function()
            .unwrap()
            .get_param_reg(1)
            .unwrap();

        // Build add instruction
        let result = builder.build_add(a, b, false).unwrap();

        // Return the result
        builder.build_return(Some(result)).unwrap();

        builder.finish_function();

        // Verify the function
        let func = &builder.module.functions[&func_id];
        assert_eq!(func.name, "add");
        assert_eq!(func.signature.parameters.len(), 2);
        assert_eq!(func.signature.return_type, IrType::I32);

        let entry_block = func.cfg.get_block(func.entry_block()).unwrap();
        assert_eq!(entry_block.instructions.len(), 1);
        assert!(matches!(
            entry_block.terminator,
            IrTerminator::Return { .. }
        ));
    }

    #[test]
    fn test_control_flow_building() {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());

        let sig = FunctionSignatureBuilder::new()
            .param("x".to_string(), IrType::I32)
            .returns(IrType::I32)
            .build();

        builder.start_function(SymbolId::from_raw(1), "abs".to_string(), sig);

        let x = builder
            .current_function()
            .unwrap()
            .get_param_reg(0)
            .unwrap();

        // Create blocks
        let negative_block = builder
            .create_block_with_label("negative".to_string())
            .unwrap();
        let positive_block = builder
            .create_block_with_label("positive".to_string())
            .unwrap();
        let merge_block = builder
            .create_block_with_label("merge".to_string())
            .unwrap();

        // Build comparison
        let zero = builder.build_int(0, IrType::I32).unwrap();
        let is_negative = builder.build_cmp(CompareOp::Lt, x, zero).unwrap();

        // Branch
        builder
            .build_cond_branch(is_negative, negative_block, positive_block)
            .unwrap();

        // Negative block
        builder.switch_to_block(negative_block);
        let neg_x = builder.build_unop(UnaryOp::Neg, x).unwrap();
        builder.build_branch(merge_block).unwrap();

        // Positive block
        builder.switch_to_block(positive_block);
        builder.build_branch(merge_block).unwrap();

        // Merge block with phi
        builder.switch_to_block(merge_block);
        let phi = builder.build_phi(merge_block, IrType::I32).unwrap();
        builder
            .add_phi_incoming(merge_block, phi, negative_block, neg_x)
            .unwrap();
        builder
            .add_phi_incoming(merge_block, phi, positive_block, x)
            .unwrap();

        builder.build_return(Some(phi)).unwrap();

        builder.finish_function();
    }
}
