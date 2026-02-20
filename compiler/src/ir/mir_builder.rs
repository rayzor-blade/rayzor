//! MIR Builder - Programmatic construction of MIR modules
//!
//! This module provides a builder API for constructing MIR (Mid-level IR)
//! programmatically, primarily used for building the standard library without
//! parsing Haxe source code.
//!
//! # Example
//!
//! ```rust,ignore
//! use compiler::ir::mir_builder::MirBuilder;
//! use compiler::ir::IrType;
//!
//! let mut builder = MirBuilder::new("MyModule");
//!
//! // Create a function: fn add(a: i32, b: i32) -> i32
//! let func_id = builder.begin_function("add")
//!     .param("a", IrType::I32)
//!     .param("b", IrType::I32)
//!     .returns(IrType::I32)
//!     .build();
//!
//! builder.set_current_function(func_id);
//!
//! // Create entry block
//! let entry = builder.create_block("entry");
//! builder.set_insert_point(entry);
//!
//! // Get parameters
//! let a = builder.get_param(0);
//! let b = builder.get_param(1);
//!
//! // Add them
//! let result = builder.bin_op(BinaryOp::Add, a, b);
//!
//! // Return result
//! builder.ret(Some(result));
//!
//! let module = builder.finish();
//! ```

use super::{
    AllocationHint, BinaryOp, CallingConvention, CompareOp, FunctionAttributes, FunctionKind,
    InlineHint, IrBasicBlock, IrBlockId, IrControlFlowGraph, IrFunction, IrFunctionId,
    IrFunctionSignature, IrId, IrInstruction, IrLocal, IrModule, IrParameter, IrSourceLocation,
    IrTerminator, IrType, IrTypeParam, IrValue, Linkage, StructField, UnaryOp, UnionVariant,
    VectorMinMaxKind, VectorUnaryOpKind,
};
use std::collections::HashMap;

/// Builder for programmatically constructing MIR modules
pub struct MirBuilder {
    /// Module being built
    module: IrModule,

    /// Current function being built
    current_function: Option<IrFunctionId>,

    /// Current block being built
    current_block: Option<IrBlockId>,

    /// Next register ID for current function
    next_reg_id: u32,
}

/// Builder for function signatures
pub struct FunctionBuilder<'a> {
    builder: &'a mut MirBuilder,
    name: String,
    params: Vec<IrParameter>,
    return_type: IrType,
    calling_convention: CallingConvention,
    can_throw: bool,
    type_params: Vec<IrTypeParam>,
    linkage: Linkage,
    inline_hint: InlineHint,
    is_extern: bool,
}

impl MirBuilder {
    /// Create a new MIR builder for a module
    pub fn new(module_name: impl Into<String>) -> Self {
        let name = module_name.into();
        let module = IrModule::new(name.clone(), format!("{}.hx", name));

        Self {
            module,
            current_function: None,
            current_block: None,
            next_reg_id: 0,
        }
    }

    /// Begin defining a new function
    pub fn begin_function(&mut self, name: impl Into<String>) -> FunctionBuilder {
        FunctionBuilder {
            builder: self,
            name: name.into(),
            params: Vec::new(),
            return_type: IrType::Void,
            calling_convention: CallingConvention::Haxe,
            can_throw: false,
            type_params: Vec::new(),
            linkage: Linkage::Public,
            inline_hint: InlineHint::Auto,
            is_extern: false,
        }
    }

    /// Set the current function being built
    pub fn set_current_function(&mut self, func_id: IrFunctionId) {
        self.current_function = Some(func_id);
        // Initialize builder's next_reg_id from the function's next_reg_id
        // (which was set to param count during build())
        let func = self
            .module
            .functions
            .get(&func_id)
            .expect("Function not found");
        self.next_reg_id = func.next_reg_id;
    }

    /// Create a new basic block in the current function
    /// If this is the first block created, it will use the existing entry block
    pub fn create_block(&mut self, label: impl Into<String>) -> IrBlockId {
        let func_id = self.current_function.expect("No current function");
        let func = self
            .module
            .functions
            .get_mut(&func_id)
            .expect("Function not found");

        // If this is the first block and the entry block exists but is unlabeled, use it
        if func.cfg.blocks.len() == 1 {
            let entry = func.cfg.entry_block;
            if let Some(block) = func.cfg.blocks.get_mut(&entry) {
                if block.label.is_none() {
                    block.label = Some(label.into());
                    return entry;
                }
            }
        }

        // Otherwise create a new block
        let block_id = IrBlockId::new(func.cfg.next_block_id);
        func.cfg.next_block_id += 1;

        let mut block = IrBasicBlock::new(block_id);
        block.label = Some(label.into());

        func.cfg.blocks.insert(block_id, block);
        block_id
    }

    /// Set the insertion point to a specific block
    pub fn set_insert_point(&mut self, block_id: IrBlockId) {
        self.current_block = Some(block_id);
    }

    /// Allocate a new register ID
    pub fn alloc_reg(&mut self) -> IrId {
        let id = IrId::new(self.next_reg_id);
        self.next_reg_id += 1;
        id
    }

    /// Allocate a new register and record its type
    fn alloc_reg_typed(&mut self, ty: IrType) -> IrId {
        let id = self.alloc_reg();
        self.register_type(id, ty);
        id
    }

    /// Register the type of a register
    fn register_type(&mut self, reg: IrId, ty: IrType) {
        let func_id = self.current_function.expect("No current function");
        let func = self
            .module
            .functions
            .get_mut(&func_id)
            .expect("Function not found");
        func.register_types.insert(reg, ty);
    }

    /// Get the type of a register
    pub fn get_register_type(&self, reg: IrId) -> Option<IrType> {
        let func_id = self.current_function?;
        let func = self.module.functions.get(&func_id)?;
        func.register_types.get(&reg).cloned()
    }

    /// Get parameter value by index
    pub fn get_param(&self, index: usize) -> IrId {
        let func_id = self.current_function.expect("No current function");
        let func = self
            .module
            .functions
            .get(&func_id)
            .expect("Function not found");
        func.signature
            .parameters
            .get(index)
            .map(|p| p.reg)
            .expect("Parameter index out of bounds")
    }

    /// Insert an instruction at the current insertion point
    fn insert_inst(&mut self, inst: IrInstruction) {
        let func_id = self.current_function.expect("No current function");
        let block_id = self.current_block.expect("No current block");

        let func = self
            .module
            .functions
            .get_mut(&func_id)
            .expect("Function not found");
        let block = func.cfg.blocks.get_mut(&block_id).expect("Block not found");

        block.instructions.push(inst);
    }

    /// Set terminator for current block
    fn set_terminator(&mut self, term: IrTerminator) {
        let func_id = self.current_function.expect("No current function");
        let block_id = self.current_block.expect("No current block");

        let func = self
            .module
            .functions
            .get_mut(&func_id)
            .expect("Function not found");
        let block = func.cfg.blocks.get_mut(&block_id).expect("Block not found");

        block.terminator = term;
    }

    // === Instruction builders ===

    /// Create a constant value
    pub fn const_value(&mut self, value: IrValue) -> IrId {
        let dest = self.alloc_reg();
        self.insert_inst(IrInstruction::Const { dest, value });
        dest
    }

    /// Create an integer constant
    pub fn const_i32(&mut self, value: i32) -> IrId {
        self.const_value(IrValue::I32(value))
    }

    /// Create a 64-bit integer constant
    pub fn const_i64(&mut self, value: i64) -> IrId {
        self.const_value(IrValue::I64(value))
    }

    /// Create a boolean constant
    pub fn const_bool(&mut self, value: bool) -> IrId {
        self.const_value(IrValue::Bool(value))
    }

    /// Create a string constant
    pub fn const_string(&mut self, value: impl Into<String>) -> IrId {
        self.const_value(IrValue::String(value.into()))
    }

    /// Create a u8 constant
    pub fn const_u8(&mut self, value: u8) -> IrId {
        self.const_value(IrValue::U8(value))
    }

    /// Create a u64 constant
    pub fn const_u64(&mut self, value: u64) -> IrId {
        self.const_value(IrValue::U64(value))
    }

    /// Load from memory
    pub fn load(&mut self, ptr: IrId, ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(ty.clone());
        self.insert_inst(IrInstruction::Load { dest, ptr, ty });
        dest
    }

    /// Store to memory
    pub fn store(&mut self, ptr: IrId, value: IrId) {
        self.insert_inst(IrInstruction::Store { ptr, value });
    }

    /// Binary operation
    pub fn bin_op(&mut self, op: BinaryOp, left: IrId, right: IrId) -> IrId {
        // Infer result type from left operand
        let ty = self.get_register_type(left).unwrap_or(IrType::I64);
        let dest = self.alloc_reg_typed(ty);
        self.insert_inst(IrInstruction::BinOp {
            dest,
            op,
            left,
            right,
        });
        dest
    }

    /// Unary operation
    pub fn un_op(&mut self, op: UnaryOp, operand: IrId) -> IrId {
        // Infer result type from operand
        let ty = self.get_register_type(operand).unwrap_or(IrType::I64);
        let dest = self.alloc_reg_typed(ty);
        self.insert_inst(IrInstruction::UnOp { dest, op, operand });
        dest
    }

    /// Compare operation
    pub fn cmp(&mut self, op: CompareOp, left: IrId, right: IrId) -> IrId {
        // Comparison always returns bool
        let dest = self.alloc_reg_typed(IrType::Bool);
        self.insert_inst(IrInstruction::Cmp {
            dest,
            op,
            left,
            right,
        });
        dest
    }

    /// Call a function directly
    pub fn call(&mut self, func_id: IrFunctionId, args: Vec<IrId>) -> Option<IrId> {
        // Clone the function signature data we need before any mutable borrows
        let (return_ty, is_c_extern, param_types) = {
            let func = self
                .module
                .functions
                .get(&func_id)
                .expect("Function not found");
            let is_extern = func.signature.calling_convention == CallingConvention::C
                && func.attributes.linkage == Linkage::External
                && !cfg!(target_os = "windows");
            let params: Vec<IrType> = func
                .signature
                .parameters
                .iter()
                .map(|p| p.ty.clone())
                .collect();
            (func.signature.return_type.clone(), is_extern, params)
        };

        let has_return = !matches!(return_ty, IrType::Void);
        let dest = if has_return {
            Some(self.alloc_reg_typed(return_ty))
        } else {
            None
        };

        // For C calling convention extern functions on non-Windows platforms,
        // we need to extend i32/u32 arguments to i64 to match the ABI.
        // However, we DON'T do this extension at the MIR level because:
        // 1. The Cranelift backend already handles ABI extension at call sites
        // 2. Doing it in both places causes double-extension errors
        // So we just pass the args as-is and let Cranelift handle the ABI.
        let adjusted_args = args;

        // Default to Move ownership for all arguments
        let arg_ownership = adjusted_args
            .iter()
            .map(|_| crate::ir::instructions::OwnershipMode::Move)
            .collect();
        self.insert_inst(IrInstruction::CallDirect {
            dest,
            func_id,
            args: adjusted_args,
            arg_ownership,
            type_args: Vec::new(),
            is_tail_call: false,
        });
        dest
    }

    /// Allocate memory
    pub fn alloc(&mut self, ty: IrType, count: Option<IrId>) -> IrId {
        // Alloc returns a pointer to the allocated type
        let ptr_ty = IrType::Ptr(Box::new(ty.clone()));
        let dest = self.alloc_reg_typed(ptr_ty);
        self.insert_inst(IrInstruction::Alloc { dest, ty, count });
        dest
    }

    /// Type cast
    pub fn cast(&mut self, src: IrId, from_ty: IrType, to_ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(to_ty.clone());
        self.insert_inst(IrInstruction::Cast {
            dest,
            src,
            from_ty,
            to_ty,
        });
        dest
    }

    /// Extract value from aggregate (struct/array) by indices
    pub fn extract_value(&mut self, aggregate: IrId, indices: Vec<u32>) -> IrId {
        // Need to infer field type from aggregate type
        // For now, use a placeholder - this should be improved
        let field_ty = self
            .get_register_type(aggregate)
            .and_then(|ty| {
                if let IrType::Struct { fields, .. } = ty {
                    if indices.len() == 1 {
                        fields.get(indices[0] as usize).map(|f| f.ty.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .unwrap_or(IrType::U64);

        let dest = self.alloc_reg_typed(field_ty);
        self.insert_inst(IrInstruction::ExtractValue {
            dest,
            aggregate,
            indices,
        });
        dest
    }

    /// Extract struct field by index
    pub fn extract_field(&mut self, aggregate: IrId, field_index: u32) -> IrId {
        self.extract_value(aggregate, vec![field_index])
    }

    /// Insert value into aggregate
    pub fn insert_value(&mut self, aggregate: IrId, value: IrId, indices: Vec<u32>) -> IrId {
        // Result has same type as aggregate
        let ty = self.get_register_type(aggregate).unwrap_or(IrType::U64);
        let dest = self.alloc_reg_typed(ty);
        self.insert_inst(IrInstruction::InsertValue {
            dest,
            aggregate,
            value,
            indices,
        });
        dest
    }

    // === Union Operations ===

    /// Create a union value with discriminant
    pub fn create_union(&mut self, discriminant: u32, value: IrId, ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(ty.clone());
        self.insert_inst(IrInstruction::CreateUnion {
            dest,
            discriminant,
            value,
            ty,
        });
        dest
    }

    /// Extract discriminant from union
    pub fn extract_discriminant(&mut self, union_val: IrId) -> IrId {
        // Discriminant is u32
        let dest = self.alloc_reg_typed(IrType::U32);
        self.insert_inst(IrInstruction::ExtractDiscriminant { dest, union_val });
        dest
    }

    /// Extract value from union variant
    pub fn extract_union_value(
        &mut self,
        union_val: IrId,
        discriminant: u32,
        value_ty: IrType,
    ) -> IrId {
        let dest = self.alloc_reg_typed(value_ty.clone());
        self.insert_inst(IrInstruction::ExtractUnionValue {
            dest,
            union_val,
            discriminant,
            value_ty,
        });
        dest
    }

    // === Struct Operations ===

    /// Create struct from field values
    pub fn create_struct(&mut self, ty: IrType, fields: Vec<IrId>) -> IrId {
        let dest = self.alloc_reg_typed(ty.clone());
        self.insert_inst(IrInstruction::CreateStruct { dest, ty, fields });
        dest
    }

    // === Pointer Operations ===

    /// Pointer arithmetic: ptr + offset
    pub fn ptr_add(&mut self, ptr: IrId, offset: IrId, ty: IrType) -> IrId {
        // Result is same pointer type
        let dest = self.alloc_reg_typed(ty.clone());
        self.insert_inst(IrInstruction::PtrAdd {
            dest,
            ptr,
            offset,
            ty,
        });
        dest
    }

    // === Comparison Operations ===

    /// Integer comparison (returns bool)
    pub fn icmp(&mut self, op: CompareOp, left: IrId, right: IrId, _result_ty: IrType) -> IrId {
        self.cmp(op, left, right)
    }

    // === Arithmetic Helper Methods ===

    /// Addition
    pub fn add(&mut self, left: IrId, right: IrId, _ty: IrType) -> IrId {
        self.bin_op(BinaryOp::Add, left, right)
    }

    /// Subtraction
    pub fn sub(&mut self, left: IrId, right: IrId, _ty: IrType) -> IrId {
        self.bin_op(BinaryOp::Sub, left, right)
    }

    /// Multiplication
    pub fn mul(&mut self, left: IrId, right: IrId, _ty: IrType) -> IrId {
        self.bin_op(BinaryOp::Mul, left, right)
    }

    // === SIMD Vector Helper Methods ===

    /// SIMD element-wise binary operation
    pub fn vector_bin_op(&mut self, op: BinaryOp, left: IrId, right: IrId, vec_ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(vec_ty.clone());
        self.insert_inst(IrInstruction::VectorBinOp {
            dest,
            op,
            left,
            right,
            vec_ty,
        });
        dest
    }

    /// Broadcast scalar to all vector lanes
    pub fn vector_splat(&mut self, scalar: IrId, vec_ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(vec_ty.clone());
        self.insert_inst(IrInstruction::VectorSplat {
            dest,
            scalar,
            vec_ty,
        });
        dest
    }

    /// Extract scalar from vector lane
    pub fn vector_extract(&mut self, vector: IrId, index: u8, elem_ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(elem_ty);
        self.insert_inst(IrInstruction::VectorExtract {
            dest,
            vector,
            index,
        });
        dest
    }

    /// Insert scalar into vector lane
    pub fn vector_insert(&mut self, vector: IrId, scalar: IrId, index: u8, vec_ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(vec_ty);
        self.insert_inst(IrInstruction::VectorInsert {
            dest,
            vector,
            scalar,
            index,
        });
        dest
    }

    /// Horizontal reduction (e.g., sum all elements)
    pub fn vector_reduce(&mut self, op: BinaryOp, vector: IrId, elem_ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(elem_ty);
        self.insert_inst(IrInstruction::VectorReduce { dest, op, vector });
        dest
    }

    /// Load contiguous elements into a SIMD vector
    pub fn vector_load(&mut self, ptr: IrId, vec_ty: IrType) -> IrId {
        let dest = self.alloc_reg_typed(vec_ty.clone());
        self.insert_inst(IrInstruction::VectorLoad { dest, ptr, vec_ty });
        dest
    }

    /// Store SIMD vector to contiguous memory
    pub fn vector_store(&mut self, ptr: IrId, value: IrId, vec_ty: IrType) {
        self.insert_inst(IrInstruction::VectorStore { ptr, value, vec_ty });
    }

    /// Element-wise unary operation on a vector (sqrt, abs, neg, ceil, floor, round)
    pub fn vector_unary_op(
        &mut self,
        op: VectorUnaryOpKind,
        operand: IrId,
        vec_ty: IrType,
    ) -> IrId {
        let dest = self.alloc_reg_typed(vec_ty.clone());
        self.insert_inst(IrInstruction::VectorUnaryOp {
            dest,
            op,
            operand,
            vec_ty,
        });
        dest
    }

    /// Element-wise min/max of two vectors
    pub fn vector_min_max(
        &mut self,
        op: VectorMinMaxKind,
        left: IrId,
        right: IrId,
        vec_ty: IrType,
    ) -> IrId {
        let dest = self.alloc_reg_typed(vec_ty.clone());
        self.insert_inst(IrInstruction::VectorMinMax {
            dest,
            op,
            left,
            right,
            vec_ty,
        });
        dest
    }

    // === Special Values ===

    /// Undefined value (uninitialized)
    pub fn undef(&mut self, ty: IrType) -> IrId {
        let dest = self.alloc_reg();
        self.insert_inst(IrInstruction::Undef { dest, ty });
        dest
    }

    /// Unit/void value
    pub fn unit_value(&mut self) -> IrId {
        self.undef(IrType::Void)
    }

    /// Function reference (for function pointers)
    pub fn function_ref(&mut self, func_id: IrFunctionId) -> IrId {
        let dest = self.alloc_reg();
        self.insert_inst(IrInstruction::FunctionRef { dest, func_id });
        dest
    }

    // === Type Construction ===

    /// Create a type parameter reference (for generic functions)
    pub fn type_param(&mut self, name: impl Into<String>) -> IrType {
        IrType::TypeVar(name.into())
    }

    /// Create a boolean type
    pub fn bool_type(&self) -> IrType {
        IrType::Bool
    }

    /// Create a u8 type
    pub fn u8_type(&self) -> IrType {
        IrType::U8
    }

    /// Create a u64/usize type
    pub fn u64_type(&self) -> IrType {
        IrType::U64
    }

    /// Create a i32 type
    pub fn i32_type(&self) -> IrType {
        IrType::I32
    }

    /// Create a void type
    pub fn void_type(&self) -> IrType {
        IrType::Void
    }

    /// Create a pointer type
    pub fn ptr_type(&self, pointee: IrType) -> IrType {
        IrType::Ptr(Box::new(pointee))
    }

    /// Create a struct type
    pub fn struct_type(&self, name: Option<impl Into<String>>, fields: Vec<IrType>) -> IrType {
        let struct_fields = fields
            .into_iter()
            .enumerate()
            .map(|(i, ty)| StructField {
                name: format!("field_{}", i),
                ty,
                offset: 0, // Will be calculated later
            })
            .collect();

        IrType::Struct {
            name: name
                .map(|n| n.into())
                .unwrap_or_else(|| String::from("anon")),
            fields: struct_fields,
        }
    }

    /// Create a union type
    pub fn union_type(
        &self,
        name: Option<impl Into<String>>,
        variants: Vec<super::UnionVariant>,
    ) -> IrType {
        IrType::Union {
            name: name
                .map(|n| n.into())
                .unwrap_or_else(|| String::from("anon")),
            variants,
        }
    }

    // === Control Flow ===

    /// Panic/abort execution
    pub fn panic(&mut self) {
        self.insert_inst(IrInstruction::Panic { message: None });
    }

    /// Unreachable terminator
    pub fn unreachable(&mut self) {
        self.set_terminator(IrTerminator::Unreachable);
    }

    // === Terminators ===

    /// Return from function
    pub fn ret(&mut self, value: Option<IrId>) {
        self.set_terminator(IrTerminator::Return { value });
    }

    /// Unconditional branch
    pub fn br(&mut self, target: IrBlockId) {
        self.set_terminator(IrTerminator::Branch { target });
    }

    /// Conditional branch
    pub fn cond_br(&mut self, condition: IrId, true_target: IrBlockId, false_target: IrBlockId) {
        self.set_terminator(IrTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        });
    }

    /// Mark a function as extern by clearing its CFG blocks, setting External linkage,
    /// and setting FunctionKind to ExternC.
    /// This is used for runtime intrinsics like malloc/realloc/free and extern C functions
    pub fn mark_as_extern(&mut self, func_id: IrFunctionId) {
        if let Some(func) = self.module.functions.get_mut(&func_id) {
            func.cfg.blocks.clear();
            func.attributes.linkage = crate::ir::Linkage::External;
            func.kind = crate::ir::functions::FunctionKind::ExternC;
        }
    }

    /// Finish building and return the module
    pub fn finish(self) -> IrModule {
        self.module
    }

    /// Get a function by name (for calling)
    pub fn get_function_by_name(&self, name: &str) -> Option<IrFunctionId> {
        self.module
            .functions
            .iter()
            .find(|(_, f)| f.name == name)
            .map(|(id, _)| *id)
    }
}

impl<'a> FunctionBuilder<'a> {
    /// Add a parameter to the function
    pub fn param(mut self, name: impl Into<String>, ty: IrType) -> Self {
        let reg = IrId::new(self.params.len() as u32);
        self.params.push(IrParameter {
            name: name.into(),
            ty,
            reg,
            by_ref: false,
        });
        self
    }

    /// Set return type
    pub fn returns(mut self, ty: IrType) -> Self {
        self.return_type = ty;
        self
    }

    /// Set calling convention
    pub fn calling_convention(mut self, cc: CallingConvention) -> Self {
        self.calling_convention = cc;
        self
    }

    /// Mark as extern function (no body, implemented in runtime)
    pub fn extern_func(mut self) -> Self {
        self.is_extern = true;
        // Extern functions have External linkage by default, but can be overridden with .public()
        if self.linkage == Linkage::Public {
            self.linkage = Linkage::External;
        }
        self
    }

    /// Mark as public
    pub fn public(mut self) -> Self {
        self.linkage = Linkage::Public;
        self
    }

    /// Set inline hint
    pub fn inline(mut self, hint: InlineHint) -> Self {
        self.inline_hint = hint;
        self
    }

    /// Build the function
    pub fn build(self) -> IrFunctionId {
        use crate::tast::SymbolId;

        let func_id = IrFunctionId(self.builder.module.next_function_id);
        self.builder.module.next_function_id += 1;

        // Initialize next_reg_id to number of parameters so we don't reuse their IDs
        let next_reg_id = self.params.len() as u32;

        // Build register_types map with parameter types BEFORE moving self.params
        let mut register_types = HashMap::new();
        for (i, param) in self.params.iter().enumerate() {
            register_types.insert(IrId(i as u32), param.ty.clone());
        }

        // Note: is_extern flag is only valid during build(), doesn't persist
        // After build(), we'll clear CFG blocks for extern functions via mark_as_extern()
        // So we can't rely on is_extern flag for sret detection later

        // For now, always set uses_sret for struct returns
        // We'll handle extern vs non-extern in the backend
        let uses_sret = matches!(&self.return_type, IrType::Struct { .. });

        let signature = IrFunctionSignature {
            parameters: self.params,
            return_type: self.return_type,
            calling_convention: self.calling_convention,
            can_throw: self.can_throw,
            type_params: self.type_params,
            uses_sret,
        };

        let mut attributes = FunctionAttributes::default();
        attributes.linkage = self.linkage;
        attributes.inline = self.inline_hint;

        // For extern functions, create an empty CFG without the default entry block
        let cfg = if self.is_extern {
            IrControlFlowGraph {
                blocks: std::collections::BTreeMap::new(),
                entry_block: IrBlockId::entry(),
                next_block_id: 0,
            }
        } else {
            IrControlFlowGraph::new()
        };

        let function = IrFunction {
            id: func_id,
            symbol_id: SymbolId::from_raw(0), // Placeholder
            name: self.name,
            qualified_name: None,
            signature,
            cfg,
            locals: HashMap::new(),
            register_types,
            attributes,
            kind: FunctionKind::MirWrapper, // MIR builder creates stdlib wrapper functions
            source_location: IrSourceLocation::unknown(),
            next_reg_id,
            type_param_tag_fixups: Vec::new(),
        };

        self.builder.module.functions.insert(func_id, function);
        func_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let mut builder = MirBuilder::new("test");

        // fn add(a: i32, b: i32) -> i32
        let func_id = builder
            .begin_function("add")
            .param("a", IrType::I32)
            .param("b", IrType::I32)
            .returns(IrType::I32)
            .build();

        builder.set_current_function(func_id);

        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);

        let a = builder.get_param(0);
        let b = builder.get_param(1);
        let result = builder.bin_op(BinaryOp::Add, a, b);
        builder.ret(Some(result));

        let module = builder.finish();

        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.name, "test");
    }
}
