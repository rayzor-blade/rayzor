//! SIMD Vectorization for MIR Optimization
//!
//! This module provides auto-vectorization passes for MIR:
//! - Loop vectorization: transform scalar loops into vector operations
//! - SLP vectorization: bundle independent scalar operations (future)
//!
//! Vectorization targets common SIMD widths (128-bit SSE/NEON, 256-bit AVX).

use super::blocks::IrTerminator;
use super::loop_analysis::{DominatorTree, LoopNestInfo, NaturalLoop, TripCount};
use super::optimization::{OptimizationPass, OptimizationResult};
use super::{
    BinaryOp, CompareOp, IrBlockId, IrControlFlowGraph, IrFunction, IrFunctionId, IrId,
    IrInstruction, IrModule, IrType, IrValue,
};
use std::collections::{HashMap, HashSet};

/// SIMD vector width in bits (target 128-bit for SSE/NEON compatibility)
pub const SIMD_WIDTH_BITS: usize = 128;

/// Vector types supported by the vectorizer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorType {
    /// 4x f32 (128-bit)
    V4F32,
    /// 2x f64 (128-bit)
    V2F64,
    /// 4x i32 (128-bit)
    V4I32,
    /// 8x i16 (128-bit)
    V8I16,
    /// 16x i8 (128-bit)
    V16I8,
}

impl VectorType {
    /// Get the scalar element type
    pub fn element_type(&self) -> IrType {
        match self {
            VectorType::V4F32 => IrType::F32,
            VectorType::V2F64 => IrType::F64,
            VectorType::V4I32 => IrType::I32,
            VectorType::V8I16 => IrType::I16,
            VectorType::V16I8 => IrType::I8,
        }
    }

    /// Get the number of elements in the vector
    pub fn num_elements(&self) -> usize {
        match self {
            VectorType::V4F32 => 4,
            VectorType::V2F64 => 2,
            VectorType::V4I32 => 4,
            VectorType::V8I16 => 8,
            VectorType::V16I8 => 16,
        }
    }

    /// Get the total size in bytes
    pub fn size_bytes(&self) -> usize {
        SIMD_WIDTH_BITS / 8
    }

    /// Get the vector type for a scalar type
    pub fn for_scalar(scalar: &IrType) -> Option<Self> {
        match scalar {
            IrType::F32 => Some(VectorType::V4F32),
            IrType::F64 => Some(VectorType::V2F64),
            IrType::I32 | IrType::U32 => Some(VectorType::V4I32),
            IrType::I16 | IrType::U16 => Some(VectorType::V8I16),
            IrType::I8 | IrType::U8 => Some(VectorType::V16I8),
            _ => None,
        }
    }

    /// Convert to IrType representation
    pub fn to_ir_type(&self) -> IrType {
        IrType::Vector {
            element: Box::new(self.element_type()),
            count: self.num_elements(),
        }
    }
}

/// SIMD vector instructions for MIR
#[derive(Debug, Clone)]
pub enum VectorInstruction {
    /// Load contiguous elements into a vector
    VectorLoad {
        dest: IrId,
        ptr: IrId,
        vec_type: VectorType,
        alignment: usize,
    },

    /// Store vector to contiguous memory
    VectorStore {
        ptr: IrId,
        value: IrId,
        vec_type: VectorType,
        alignment: usize,
    },

    /// Vector binary operation (element-wise)
    VectorBinOp {
        dest: IrId,
        op: BinaryOp,
        left: IrId,
        right: IrId,
        vec_type: VectorType,
    },

    /// Broadcast scalar to all vector lanes
    VectorSplat {
        dest: IrId,
        scalar: IrId,
        vec_type: VectorType,
    },

    /// Extract scalar element from vector
    VectorExtract {
        dest: IrId,
        vector: IrId,
        index: usize,
        vec_type: VectorType,
    },

    /// Insert scalar into vector lane
    VectorInsert {
        dest: IrId,
        vector: IrId,
        scalar: IrId,
        index: usize,
        vec_type: VectorType,
    },

    /// Horizontal reduction (e.g., sum all elements)
    VectorReduce {
        dest: IrId,
        op: BinaryOp,
        vector: IrId,
        vec_type: VectorType,
    },

    /// Vector comparison (produces mask)
    VectorCmp {
        dest: IrId,
        op: CompareOp,
        left: IrId,
        right: IrId,
        vec_type: VectorType,
    },

    /// Masked load (load where mask is true)
    VectorMaskedLoad {
        dest: IrId,
        ptr: IrId,
        mask: IrId,
        passthru: IrId,
        vec_type: VectorType,
    },

    /// Masked store (store where mask is true)
    VectorMaskedStore {
        ptr: IrId,
        value: IrId,
        mask: IrId,
        vec_type: VectorType,
    },

    /// Gather (indexed load)
    VectorGather {
        dest: IrId,
        base: IrId,
        indices: IrId,
        mask: IrId,
        vec_type: VectorType,
    },

    /// Scatter (indexed store)
    VectorScatter {
        base: IrId,
        indices: IrId,
        value: IrId,
        mask: IrId,
        vec_type: VectorType,
    },
}

/// Analysis result for a loop's vectorizability
#[derive(Debug)]
pub struct VectorizationAnalysis {
    /// Whether the loop can be vectorized
    pub can_vectorize: bool,

    /// Reason if vectorization is not possible
    pub failure_reason: Option<String>,

    /// The vector width to use (number of elements per iteration)
    pub vector_factor: usize,

    /// Scalar type being vectorized
    pub scalar_type: Option<IrType>,

    /// Induction variable (loop counter)
    pub induction_var: Option<IrId>,

    /// Memory accesses that can be vectorized
    pub vectorizable_accesses: Vec<MemoryAccess>,

    /// Reductions that can be vectorized
    pub reductions: Vec<Reduction>,

    /// Instructions that must remain scalar (e.g., loop control)
    pub scalar_instructions: HashSet<IrId>,

    /// Estimated speedup from vectorization
    pub estimated_speedup: f64,
}

/// A memory access pattern in a loop
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    /// The instruction ID
    pub instruction_id: usize,

    /// Base pointer
    pub base: IrId,

    /// Stride (elements per iteration, 1 = contiguous)
    pub stride: i64,

    /// Is this a load or store
    pub is_load: bool,

    /// Element type
    pub element_type: IrType,
}

/// A reduction operation in a loop
#[derive(Debug, Clone)]
pub struct Reduction {
    /// The accumulator variable
    pub accumulator: IrId,

    /// The reduction operation
    pub op: BinaryOp,

    /// Initial value
    pub init_value: IrValue,

    /// The instruction performing the reduction
    pub instruction_id: usize,
}

/// Loop Vectorization Pass
///
/// Transforms scalar loops into vector operations where profitable.
/// Uses the following strategy:
/// 1. Analyze loop for vectorizability
/// 2. Determine optimal vector factor
/// 3. Generate vector loop body
/// 4. Generate epilog for remainder iterations
pub struct LoopVectorizationPass {
    /// Minimum trip count for vectorization (skip small loops)
    pub min_trip_count: usize,

    /// Enable cost-model based decisions
    pub use_cost_model: bool,

    /// Target vector width in bits
    pub target_width: usize,
}

impl Default for LoopVectorizationPass {
    fn default() -> Self {
        Self {
            min_trip_count: 8,
            use_cost_model: true,
            target_width: SIMD_WIDTH_BITS,
        }
    }
}

impl LoopVectorizationPass {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze a loop for vectorization potential
    pub fn analyze_loop(
        &self,
        function: &IrFunction,
        loop_info: &NaturalLoop,
        domtree: &DominatorTree,
    ) -> VectorizationAnalysis {
        let mut analysis = VectorizationAnalysis {
            can_vectorize: false,
            failure_reason: None,
            vector_factor: 1,
            scalar_type: None,
            induction_var: None,
            vectorizable_accesses: Vec::new(),
            reductions: Vec::new(),
            scalar_instructions: HashSet::new(),
            estimated_speedup: 1.0,
        };

        // Check 1: Loop must have a single exit
        if loop_info.exit_blocks.len() != 1 {
            analysis.failure_reason = Some("Loop has multiple exits".to_string());
            return analysis;
        }

        // Check 2: Check trip count
        match &loop_info.trip_count {
            Some(TripCount::Constant(n)) if *n < self.min_trip_count as u64 => {
                analysis.failure_reason = Some(format!(
                    "Trip count {} too small (min: {})",
                    n, self.min_trip_count
                ));
                return analysis;
            }
            None => {
                // Unknown trip count - can still vectorize with runtime check
            }
            _ => {}
        }

        // Check 3: Find induction variable
        let induction_var = self.find_induction_variable(function, loop_info);
        if induction_var.is_none() {
            analysis.failure_reason = Some("No suitable induction variable found".to_string());
            return analysis;
        }
        analysis.induction_var = induction_var;

        // Check 4: Analyze memory accesses
        let memory_accesses = self.analyze_memory_accesses(function, loop_info);

        // Filter for contiguous, vectorizable accesses
        let vectorizable: Vec<_> = memory_accesses
            .into_iter()
            .filter(|acc| acc.stride == 1 && VectorType::for_scalar(&acc.element_type).is_some())
            .collect();

        if vectorizable.is_empty() {
            analysis.failure_reason = Some("No vectorizable memory accesses".to_string());
            return analysis;
        }

        // Determine the dominant scalar type
        let scalar_type = vectorizable
            .first()
            .map(|acc| acc.element_type.clone())
            .unwrap();

        let vector_type = VectorType::for_scalar(&scalar_type);
        if vector_type.is_none() {
            analysis.failure_reason = Some("Unsupported element type".to_string());
            return analysis;
        }

        let vec_type = vector_type.unwrap();
        analysis.scalar_type = Some(scalar_type);
        analysis.vector_factor = vec_type.num_elements();
        analysis.vectorizable_accesses = vectorizable;

        // Check 5: Analyze reductions
        analysis.reductions = self.find_reductions(function, loop_info);

        // Check 6: Check for unsupported operations
        if let Some(reason) = self.check_unsupported_ops(function, loop_info) {
            analysis.failure_reason = Some(reason);
            return analysis;
        }

        // Cost model check
        if self.use_cost_model {
            let speedup = self.estimate_speedup(&analysis, loop_info);
            analysis.estimated_speedup = speedup;

            if speedup < 1.5 {
                analysis.failure_reason =
                    Some(format!("Estimated speedup {:.2}x too low", speedup));
                return analysis;
            }
        }

        analysis.can_vectorize = true;
        analysis
    }

    /// Find the loop's induction variable (typically i in `for i = 0 to n`)
    fn find_induction_variable(
        &self,
        function: &IrFunction,
        loop_info: &NaturalLoop,
    ) -> Option<IrId> {
        let header = loop_info.header;
        let header_block = function.cfg.blocks.get(&header)?;

        // Look for phi nodes in the header that:
        // 1. Have one incoming value from outside the loop (initial value)
        // 2. Have one incoming value from inside the loop (increment)
        // 3. The increment is a simple add/sub by constant

        for inst in &header_block.instructions {
            if let IrInstruction::Phi { dest, incoming } = inst {
                // Check if this phi is an induction variable
                let mut outside_val = None;
                let mut inside_val = None;

                for (val, pred) in incoming {
                    // Convert IrId to IrBlockId for comparison with loop blocks
                    let pred_block = IrBlockId::new(pred.as_u32());
                    if loop_info.blocks.contains(&pred_block) {
                        inside_val = Some(*val);
                    } else {
                        outside_val = Some(*val);
                    }
                }

                // Need both an outside (init) and inside (update) value
                if outside_val.is_some() && inside_val.is_some() {
                    // Check if the inside value is dest + constant
                    if self.is_simple_increment(function, loop_info, *dest, inside_val.unwrap()) {
                        return Some(*dest);
                    }
                }
            }
        }

        None
    }

    /// Check if `updated` is `original + constant` within the loop
    fn is_simple_increment(
        &self,
        function: &IrFunction,
        loop_info: &NaturalLoop,
        original: IrId,
        updated: IrId,
    ) -> bool {
        // Search for the instruction that defines `updated`
        for block_id in &loop_info.blocks {
            if let Some(block) = function.cfg.blocks.get(block_id) {
                for inst in &block.instructions {
                    if let IrInstruction::BinOp {
                        dest,
                        op: BinaryOp::Add | BinaryOp::Sub,
                        left,
                        right,
                    } = inst
                    {
                        if *dest == updated && (*left == original || *right == original) {
                            // Check if the other operand is a constant
                            // For now, accept any simple add/sub
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Analyze memory accesses in the loop
    fn analyze_memory_accesses(
        &self,
        function: &IrFunction,
        loop_info: &NaturalLoop,
    ) -> Vec<MemoryAccess> {
        let mut accesses = Vec::new();
        let mut inst_idx = 0;

        for block_id in &loop_info.blocks {
            if let Some(block) = function.cfg.blocks.get(block_id) {
                for inst in &block.instructions {
                    match inst {
                        IrInstruction::Load { dest: _, ptr, ty } => {
                            accesses.push(MemoryAccess {
                                instruction_id: inst_idx,
                                base: *ptr,
                                stride: 1, // Assume unit stride, could analyze further
                                is_load: true,
                                element_type: ty.clone(),
                            });
                        }
                        IrInstruction::Store { ptr, value: _ } => {
                            // Need to determine the type from context
                            accesses.push(MemoryAccess {
                                instruction_id: inst_idx,
                                base: *ptr,
                                stride: 1,
                                is_load: false,
                                element_type: IrType::I64, // Placeholder
                            });
                        }
                        _ => {}
                    }
                    inst_idx += 1;
                }
            }
        }

        accesses
    }

    /// Find reduction patterns in the loop
    fn find_reductions(&self, function: &IrFunction, loop_info: &NaturalLoop) -> Vec<Reduction> {
        let mut reductions = Vec::new();
        let header = loop_info.header;

        if let Some(header_block) = function.cfg.blocks.get(&header) {
            for inst in &header_block.instructions {
                if let IrInstruction::Phi { dest, incoming } = inst {
                    // Check if this phi accumulates via associative op
                    if let Some(reduction) =
                        self.check_reduction_phi(function, loop_info, *dest, incoming)
                    {
                        reductions.push(reduction);
                    }
                }
            }
        }

        reductions
    }

    /// Check if a phi node represents a reduction
    fn check_reduction_phi(
        &self,
        function: &IrFunction,
        loop_info: &NaturalLoop,
        phi_dest: IrId,
        incoming: &[(IrId, IrId)],
    ) -> Option<Reduction> {
        // Find the in-loop update
        let mut in_loop_val = None;
        let mut init_val = None;

        for (val, pred) in incoming {
            // Convert IrId to IrBlockId for comparison with loop blocks
            let pred_block = IrBlockId::new(pred.as_u32());
            if loop_info.blocks.contains(&pred_block) {
                in_loop_val = Some(*val);
            } else {
                init_val = Some(*val);
            }
        }

        let updated_val = in_loop_val?;
        let _init = init_val?;

        // Find the instruction that computes the update
        for block_id in &loop_info.blocks {
            if let Some(block) = function.cfg.blocks.get(block_id) {
                for (idx, inst) in block.instructions.iter().enumerate() {
                    if let IrInstruction::BinOp {
                        dest,
                        op,
                        left,
                        right,
                    } = inst
                    {
                        if *dest == updated_val
                            && (*left == phi_dest || *right == phi_dest)
                            && Self::is_reduction_op(*op)
                        {
                            return Some(Reduction {
                                accumulator: phi_dest,
                                op: *op,
                                init_value: IrValue::I64(0), // Would need to look up actual init
                                instruction_id: idx,
                            });
                        }
                    }
                }
            }
        }

        None
    }

    /// Check if an operation can be used for reduction
    fn is_reduction_op(op: BinaryOp) -> bool {
        matches!(
            op,
            BinaryOp::Add | BinaryOp::Mul | BinaryOp::And | BinaryOp::Or | BinaryOp::Xor
        )
    }

    /// Check for operations that prevent vectorization
    fn check_unsupported_ops(
        &self,
        function: &IrFunction,
        loop_info: &NaturalLoop,
    ) -> Option<String> {
        for block_id in &loop_info.blocks {
            if let Some(block) = function.cfg.blocks.get(block_id) {
                for inst in &block.instructions {
                    match inst {
                        // Function calls might have side effects
                        IrInstruction::CallDirect { .. } | IrInstruction::CallIndirect { .. } => {
                            return Some("Loop contains function calls".to_string());
                        }
                        // Exceptions break vectorization
                        IrInstruction::Throw { .. } => {
                            return Some("Loop contains exception handling".to_string());
                        }
                        // Division needs special handling for div-by-zero
                        IrInstruction::BinOp {
                            op: BinaryOp::Div | BinaryOp::Rem,
                            ..
                        } => {
                            return Some("Loop contains division".to_string());
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }

    /// Estimate the speedup from vectorization
    fn estimate_speedup(&self, analysis: &VectorizationAnalysis, loop_info: &NaturalLoop) -> f64 {
        // Simple cost model:
        // - Each vectorized memory op saves (VF-1) ops
        // - Each vectorized arithmetic op saves (VF-1) ops
        // - Overhead: prologue, epilogue, reduction finalization

        let vf = analysis.vector_factor as f64;
        let num_mem_ops = analysis.vectorizable_accesses.len() as f64;
        let num_reductions = analysis.reductions.len() as f64;

        // Estimate loop body arithmetic operations
        let estimated_arith_ops = loop_info.blocks.len() as f64 * 2.0;

        // Vector speedup (idealized)
        let vector_benefit = (num_mem_ops + estimated_arith_ops) * (vf - 1.0) / vf;

        // Overhead costs
        let overhead = 2.0 + num_reductions * 2.0; // Prologue + epilogue + reduction finalize

        // Trip count factor
        let trip_factor = match &loop_info.trip_count {
            Some(TripCount::Constant(n)) => (*n as f64 / vf).max(1.0),
            _ => 10.0, // Assume reasonable trip count
        };

        let speedup = (vector_benefit * trip_factor) / (trip_factor + overhead);
        speedup.max(0.1) // Floor at 0.1 to avoid negative/zero
    }

    /// Transform a loop to use vector operations
    ///
    /// This performs the actual loop vectorization transformation:
    /// 1. Validates vectorization prerequisites
    /// 2. Replaces scalar operations with SIMD vector operations in-place
    /// 3. Updates the induction variable stride from 1 to VF
    /// 4. Adjusts loop bounds for vector iterations
    /// 5. Creates epilogue for remainder iterations (when trip_count % VF != 0)
    pub fn vectorize_loop(
        &self,
        function: &mut IrFunction,
        loop_info: &NaturalLoop,
        analysis: &VectorizationAnalysis,
    ) -> bool {
        if !analysis.can_vectorize {
            return false;
        }

        let vf = analysis.vector_factor;
        let induction_var = match analysis.induction_var {
            Some(iv) => iv,
            None => return false,
        };

        let scalar_type = match &analysis.scalar_type {
            Some(ty) => ty.clone(),
            None => return false,
        };

        let vec_type = match VectorType::for_scalar(&scalar_type) {
            Some(vt) => vt,
            None => return false,
        };

        // For constant trip counts, we can directly vectorize
        // For bounded/symbolic/unknown, we would need runtime checks which aren't implemented yet
        let trip_count = match &loop_info.trip_count {
            Some(TripCount::Constant(n)) if *n >= vf as u64 => *n,
            Some(TripCount::Constant(_)) => return false, // Too small to vectorize
            Some(TripCount::Bounded { max }) if *max >= vf as u64 => {
                // For bounded trip counts, we could use runtime checks, but for now skip
                return false;
            }
            Some(TripCount::Bounded { .. }) => return false, // Too small
            Some(TripCount::Symbolic { .. }) => return false, // Would need runtime iteration count
            Some(TripCount::Unknown) => return false,        // Cannot vectorize without trip count
            None => return false,                            // No trip count analysis available
        };

        let vector_iterations = trip_count / vf as u64;
        let remainder = trip_count % vf as u64;

        // Transform each block in the loop
        for block_id in &loop_info.blocks {
            if let Some(block) = function.cfg.blocks.get_mut(block_id) {
                let mut vectorized_instructions = Vec::with_capacity(block.instructions.len());

                for inst in &block.instructions {
                    let vectorized = self.vectorize_instruction(
                        inst,
                        &analysis.vectorizable_accesses,
                        &vec_type,
                        induction_var,
                        vf,
                    );
                    vectorized_instructions.push(vectorized);
                }

                // Replace the block's instructions with vectorized versions
                block.instructions = vectorized_instructions;
            }
        }

        // Update the induction variable's increment from 1 to VF
        self.update_induction_stride(function, loop_info, induction_var, vf);

        // Update loop bound comparison (divide by VF)
        self.update_loop_bound(function, loop_info, vector_iterations, vf);

        // Create epilogue loop for remainder iterations if needed
        if remainder > 0 {
            self.create_epilogue_loop(function, loop_info, remainder as usize, &scalar_type);
        }

        // Handle reductions - finalize vector reductions to scalar
        for reduction in &analysis.reductions {
            self.finalize_reduction(function, loop_info, reduction, &vec_type);
        }

        true
    }

    /// Vectorize a single instruction
    fn vectorize_instruction(
        &self,
        inst: &IrInstruction,
        vectorizable_accesses: &[MemoryAccess],
        vec_type: &VectorType,
        induction_var: IrId,
        vf: usize,
    ) -> IrInstruction {
        match inst {
            // Transform contiguous loads to vector loads
            IrInstruction::Load { dest, ptr, ty }
                if self.is_vectorizable_access(*ptr, vectorizable_accesses) =>
            {
                IrInstruction::VectorLoad {
                    dest: *dest,
                    ptr: *ptr,
                    vec_ty: vec_type.to_ir_type(),
                }
            }

            // Transform contiguous stores to vector stores
            IrInstruction::Store { ptr, value }
                if self.is_vectorizable_access(*ptr, vectorizable_accesses) =>
            {
                IrInstruction::VectorStore {
                    ptr: *ptr,
                    value: *value,
                    vec_ty: vec_type.to_ir_type(),
                }
            }

            // Transform vectorizable binary operations
            IrInstruction::BinOp {
                dest,
                op,
                left,
                right,
            } if Self::is_vectorizable_binop(*op) => IrInstruction::VectorBinOp {
                dest: *dest,
                op: *op,
                left: *left,
                right: *right,
                vec_ty: vec_type.to_ir_type(),
            },

            // Keep non-vectorizable instructions unchanged
            other => other.clone(),
        }
    }

    /// Update the induction variable's stride from 1 to VF.
    ///
    /// Finds `iv_next = iv + 1` and rewrites to `iv_next = iv + VF` by inserting
    /// a new constant register and updating the BinOp operand.
    fn update_induction_stride(
        &self,
        function: &mut IrFunction,
        loop_info: &NaturalLoop,
        induction_var: IrId,
        vf: usize,
    ) {
        // Allocate a fresh register for the VF constant
        let vf_reg = IrId::new(function.next_reg_id);
        function.next_reg_id += 1;
        function.register_types.insert(vf_reg, IrType::I32);

        for block_id in &loop_info.blocks {
            if let Some(block) = function.cfg.blocks.get_mut(block_id) {
                let mut insert_const_before = None;

                for (idx, inst) in block.instructions.iter_mut().enumerate() {
                    if let IrInstruction::BinOp {
                        op: BinaryOp::Add,
                        left,
                        right,
                        ..
                    } = inst
                    {
                        if *left == induction_var || *right == induction_var {
                            // Replace the stride operand with VF
                            if *left == induction_var {
                                *right = vf_reg;
                            } else {
                                *left = vf_reg;
                            }
                            insert_const_before = Some(idx);
                            break;
                        }
                    }
                }

                // Insert the VF constant before the updated BinOp
                if let Some(idx) = insert_const_before {
                    block.instructions.insert(
                        idx,
                        IrInstruction::Const {
                            dest: vf_reg,
                            value: IrValue::I32(vf as i32),
                        },
                    );
                    return; // Done — one IV increment per loop
                }
            }
        }
    }

    /// Update the loop bound for vector iterations.
    ///
    /// The original `i < N` becomes `i < vector_iterations * VF` so the vector
    /// loop exits before the remainder (handled by the epilogue).
    fn update_loop_bound(
        &self,
        function: &mut IrFunction,
        loop_info: &NaturalLoop,
        vector_iterations: u64,
        vf: usize,
    ) {
        let adjusted_bound = (vector_iterations * vf as u64) as i32;

        // Allocate a fresh register for the adjusted bound constant
        let bound_reg = IrId::new(function.next_reg_id);
        function.next_reg_id += 1;
        function.register_types.insert(bound_reg, IrType::I32);

        // Find the comparison instruction in the header or latch block
        // Typically the Cmp is in the header block (condition check)
        let header = loop_info.header;
        let blocks_to_check: Vec<IrBlockId> = std::iter::once(header)
            .chain(loop_info.blocks.iter().copied())
            .collect();

        for block_id in blocks_to_check {
            if let Some(block) = function.cfg.blocks.get_mut(&block_id) {
                let mut insert_const_before = None;

                for (idx, inst) in block.instructions.iter_mut().enumerate() {
                    if let IrInstruction::Cmp {
                        op: CompareOp::Lt | CompareOp::Le,
                        right,
                        ..
                    } = inst
                    {
                        // Replace the bound operand
                        *right = bound_reg;
                        insert_const_before = Some(idx);
                        break;
                    }
                }

                if let Some(idx) = insert_const_before {
                    block.instructions.insert(
                        idx,
                        IrInstruction::Const {
                            dest: bound_reg,
                            value: IrValue::I32(adjusted_bound),
                        },
                    );
                    return;
                }
            }
        }
    }

    /// Create an epilogue for remainder iterations after the vector loop.
    ///
    /// For constant trip counts the remainder is known at compile time.
    /// We collect the original scalar body instructions (pre-vectorization is too
    /// late — they've been rewritten), so instead we emit scalar copies of the
    /// vectorized ops by reversing the vector→scalar mapping.  For simplicity we
    /// unroll small remainders (≤ VF, which is always the case for constant trip
    /// counts with 128-bit vectors) into the exit block.
    fn create_epilogue_loop(
        &self,
        function: &mut IrFunction,
        loop_info: &NaturalLoop,
        remainder: usize,
        scalar_type: &IrType,
    ) {
        // The exit block is where the epilogue goes
        let exit_block_id = match loop_info.exit_blocks.first() {
            Some(id) => *id,
            None => return,
        };

        // Collect scalar instructions from the loop body blocks (excluding the
        // header, which contains phi/cmp/branch — not data work).  We reverse
        // vector instructions back to scalar form.
        let mut scalar_body: Vec<IrInstruction> = Vec::new();
        for block_id in &loop_info.blocks {
            if *block_id == loop_info.header {
                continue;
            }
            if let Some(block) = function.cfg.blocks.get(block_id) {
                for inst in &block.instructions {
                    if let Some(scalar_inst) = self.devectorize_instruction(inst, scalar_type) {
                        scalar_body.push(scalar_inst);
                    }
                }
            }
        }

        if scalar_body.is_empty() {
            return;
        }

        // Unroll remainder iterations into a new epilogue block
        let epilogue_block_id = function.cfg.create_block();
        let mut epilogue_instructions = Vec::new();
        let mut reg_id = function.next_reg_id;

        for iteration in 0..remainder {
            let mut reg_map: HashMap<IrId, IrId> = HashMap::new();

            for inst in &scalar_body {
                let new_inst =
                    self.remap_epilogue_instruction(inst, &mut reg_map, &mut reg_id, function);
                epilogue_instructions.push(new_inst);
            }

            let _ = iteration; // Each iteration is independent for contiguous access
        }

        function.next_reg_id = reg_id;

        // Wire epilogue: vector loop exit → epilogue → original exit target
        // The vector loop's exit block currently branches to the original exit.
        // We intercept: exit block → epilogue block → original successor.
        if let Some(epilogue_block) = function.cfg.blocks.get_mut(&epilogue_block_id) {
            epilogue_block.instructions = epilogue_instructions;
            // Branch to the original exit block
            epilogue_block.terminator = IrTerminator::Branch {
                target: exit_block_id,
            };
            epilogue_block.predecessors = vec![loop_info.header];
        }

        // Re-route: the loop exit (header's false branch) now goes to epilogue
        // instead of directly to the exit block.
        if let Some(header_block) = function.cfg.blocks.get_mut(&loop_info.header) {
            Self::replace_terminator_target(
                &mut header_block.terminator,
                exit_block_id,
                epilogue_block_id,
            );
        }

        // Update exit block predecessors
        if let Some(exit_block) = function.cfg.blocks.get_mut(&exit_block_id) {
            for pred in &mut exit_block.predecessors {
                if *pred == loop_info.header {
                    *pred = epilogue_block_id;
                }
            }
        }
    }

    /// Convert a vector instruction back to its scalar equivalent for epilogue.
    fn devectorize_instruction(
        &self,
        inst: &IrInstruction,
        scalar_type: &IrType,
    ) -> Option<IrInstruction> {
        match inst {
            IrInstruction::VectorLoad { dest, ptr, .. } => Some(IrInstruction::Load {
                dest: *dest,
                ptr: *ptr,
                ty: scalar_type.clone(),
            }),
            IrInstruction::VectorStore { ptr, value, .. } => Some(IrInstruction::Store {
                ptr: *ptr,
                value: *value,
            }),
            IrInstruction::VectorBinOp {
                dest,
                op,
                left,
                right,
                ..
            } => Some(IrInstruction::BinOp {
                dest: *dest,
                op: *op,
                left: *left,
                right: *right,
            }),
            // Skip non-data instructions (consts, GEPs for IV, etc.)
            IrInstruction::Const { .. }
            | IrInstruction::BinOp { .. }
            | IrInstruction::GetElementPtr { .. } => Some(inst.clone()),
            _ => None,
        }
    }

    /// Clone an instruction with fresh registers for epilogue unrolling.
    fn remap_epilogue_instruction(
        &self,
        inst: &IrInstruction,
        reg_map: &mut HashMap<IrId, IrId>,
        next_reg: &mut u32,
        function: &mut IrFunction,
    ) -> IrInstruction {
        let map_use =
            |r: IrId, map: &HashMap<IrId, IrId>| -> IrId { map.get(&r).copied().unwrap_or(r) };
        let alloc_new = |old: IrId,
                         next: &mut u32,
                         map: &mut HashMap<IrId, IrId>,
                         func: &mut IrFunction|
         -> IrId {
            if let Some(&existing) = map.get(&old) {
                return existing;
            }
            let new = IrId::new(*next);
            *next += 1;
            map.insert(old, new);
            if let Some(ty) = func.register_types.get(&old).cloned() {
                func.register_types.insert(new, ty);
            }
            new
        };

        match inst {
            IrInstruction::Const { dest, value } => IrInstruction::Const {
                dest: alloc_new(*dest, next_reg, reg_map, function),
                value: value.clone(),
            },
            IrInstruction::Load { dest, ptr, ty } => IrInstruction::Load {
                dest: alloc_new(*dest, next_reg, reg_map, function),
                ptr: map_use(*ptr, reg_map),
                ty: ty.clone(),
            },
            IrInstruction::Store { ptr, value } => IrInstruction::Store {
                ptr: map_use(*ptr, reg_map),
                value: map_use(*value, reg_map),
            },
            IrInstruction::BinOp {
                dest,
                op,
                left,
                right,
            } => IrInstruction::BinOp {
                dest: alloc_new(*dest, next_reg, reg_map, function),
                op: *op,
                left: map_use(*left, reg_map),
                right: map_use(*right, reg_map),
            },
            IrInstruction::GetElementPtr {
                dest,
                ptr,
                indices,
                ty,
            } => IrInstruction::GetElementPtr {
                dest: alloc_new(*dest, next_reg, reg_map, function),
                ptr: map_use(*ptr, reg_map),
                indices: indices.iter().map(|i| map_use(*i, reg_map)).collect(),
                ty: ty.clone(),
            },
            other => other.clone(),
        }
    }

    /// Replace a specific target in a block terminator.
    fn replace_terminator_target(
        terminator: &mut IrTerminator,
        old_target: IrBlockId,
        new_target: IrBlockId,
    ) {
        match terminator {
            IrTerminator::Branch { target } if *target == old_target => {
                *target = new_target;
            }
            IrTerminator::CondBranch {
                true_target,
                false_target,
                ..
            } => {
                if *true_target == old_target {
                    *true_target = new_target;
                }
                if *false_target == old_target {
                    *false_target = new_target;
                }
            }
            _ => {}
        }
    }

    /// Finalize a reduction after the vector loop
    fn finalize_reduction(
        &self,
        function: &mut IrFunction,
        loop_info: &NaturalLoop,
        reduction: &Reduction,
        vec_type: &VectorType,
    ) {
        // After the vector loop, we need to reduce the vector accumulator to a scalar
        // Insert a VectorReduce instruction after the loop
        if let Some(exit_block) = loop_info.exit_blocks.first() {
            if let Some(block) = function.cfg.blocks.get_mut(exit_block) {
                // Insert reduction finalization at the start of the exit block
                let reduce_inst = IrInstruction::VectorReduce {
                    dest: reduction.accumulator,
                    op: reduction.op,
                    vector: reduction.accumulator,
                };
                block.instructions.insert(0, reduce_inst);
                let _ = vec_type; // Used for type checking in full implementation
            }
        }
    }

    /// Check if a memory access is in the vectorizable list
    fn is_vectorizable_access(&self, ptr: IrId, accesses: &[MemoryAccess]) -> bool {
        accesses.iter().any(|acc| acc.base == ptr)
    }

    /// Check if a binary operation can be vectorized
    fn is_vectorizable_binop(op: BinaryOp) -> bool {
        matches!(
            op,
            BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::FAdd
                | BinaryOp::FSub
                | BinaryOp::FMul
                | BinaryOp::FDiv
                | BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Xor
        )
    }
}

impl OptimizationPass for LoopVectorizationPass {
    fn name(&self) -> &'static str {
        "LoopVectorization"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            let func_result = self.run_on_function(function);
            result = result.combine(func_result);
        }

        result
    }

    fn run_on_function(&mut self, function: &mut IrFunction) -> OptimizationResult {
        let domtree = DominatorTree::compute(function);
        let loop_nest = LoopNestInfo::analyze(function, &domtree);

        let mut modified = false;

        // Process innermost loops first (they're most likely to benefit)
        for loop_info in loop_nest.loops_innermost_first() {
            let analysis = self.analyze_loop(function, loop_info, &domtree);

            if analysis.can_vectorize {
                if self.vectorize_loop(function, loop_info, &analysis) {
                    modified = true;
                }
            }
        }

        if modified {
            OptimizationResult::changed()
        } else {
            OptimizationResult::unchanged()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::blocks::IrTerminator;
    use crate::ir::functions::IrFunctionSignature;
    use crate::tast::SymbolId;

    fn make_test_function() -> IrFunction {
        IrFunction::new(
            IrFunctionId(0),
            SymbolId::from_raw(0),
            "test".to_string(),
            IrFunctionSignature {
                parameters: vec![],
                return_type: IrType::Void,
                calling_convention: crate::ir::CallingConvention::Haxe,
                can_throw: false,
                type_params: vec![],
                uses_sret: false,
            },
        )
    }

    fn make_loop_info(
        header: IrBlockId,
        body: IrBlockId,
        exit: IrBlockId,
        trip_count: u64,
    ) -> NaturalLoop {
        NaturalLoop {
            header,
            back_edge_source: body,
            blocks: [header, body].into_iter().collect(),
            exit_blocks: vec![exit],
            preheader: None,
            trip_count: Some(TripCount::Constant(trip_count)),
            nesting_depth: 0,
            parent: None,
            children: vec![],
        }
    }

    #[test]
    fn test_vector_types() {
        assert_eq!(VectorType::V4F32.num_elements(), 4);
        assert_eq!(VectorType::V2F64.num_elements(), 2);
        assert_eq!(VectorType::V4F32.element_type(), IrType::F32);
        assert_eq!(VectorType::V4F32.size_bytes(), 16);

        assert_eq!(
            VectorType::for_scalar(&IrType::F32),
            Some(VectorType::V4F32)
        );
        assert_eq!(
            VectorType::for_scalar(&IrType::F64),
            Some(VectorType::V2F64)
        );
        assert_eq!(VectorType::for_scalar(&IrType::Bool), None);
    }

    #[test]
    fn test_update_induction_stride() {
        // Build a minimal loop: header has phi + cmp, body has iv + 1
        let mut function = make_test_function();

        let header = function.cfg.entry_block;
        let body = function.cfg.create_block();
        let exit = function.cfg.create_block();

        let iv = IrId::new(0); // induction variable
        let iv_init = IrId::new(1); // init value (from preheader)
        let iv_next = IrId::new(2); // updated IV
        let one = IrId::new(3); // constant 1
        let cmp_result = IrId::new(4);
        let bound = IrId::new(5);
        let ptr = IrId::new(6);
        let loaded = IrId::new(7);
        function.next_reg_id = 8;

        // Header: phi(iv) + cmp + condbranch
        if let Some(hdr) = function.cfg.blocks.get_mut(&header) {
            hdr.instructions = vec![
                IrInstruction::Phi {
                    dest: iv,
                    incoming: vec![
                        (iv_init, IrId::new(99)), // from "preheader" (fake)
                        (iv_next, IrId::new(body.as_u32())),
                    ],
                },
                IrInstruction::Cmp {
                    dest: cmp_result,
                    op: CompareOp::Lt,
                    left: iv,
                    right: bound,
                },
            ];
            hdr.terminator = IrTerminator::CondBranch {
                condition: cmp_result,
                true_target: body,
                false_target: exit,
            };
        }

        // Body: load + iv+1 + branch back
        if let Some(b) = function.cfg.blocks.get_mut(&body) {
            b.instructions = vec![
                IrInstruction::Load {
                    dest: loaded,
                    ptr,
                    ty: IrType::F32,
                },
                IrInstruction::Const {
                    dest: one,
                    value: IrValue::I32(1),
                },
                IrInstruction::BinOp {
                    dest: iv_next,
                    op: BinaryOp::Add,
                    left: iv,
                    right: one,
                },
            ];
            b.terminator = IrTerminator::Branch { target: header };
        }

        // Exit: return
        if let Some(e) = function.cfg.blocks.get_mut(&exit) {
            e.terminator = IrTerminator::Return { value: None };
        }

        let loop_info = make_loop_info(header, body, exit, 16);

        let pass = LoopVectorizationPass::new();
        pass.update_induction_stride(&mut function, &loop_info, iv, 4);

        // The body should now have a new Const(4) and the BinOp should use it
        let body_block = function.cfg.blocks.get(&body).unwrap();
        let mut found_vf_const = false;
        let mut stride_updated = false;

        for inst in &body_block.instructions {
            if let IrInstruction::Const {
                value: IrValue::I32(4),
                ..
            } = inst
            {
                found_vf_const = true;
            }
            if let IrInstruction::BinOp {
                op: BinaryOp::Add,
                left,
                right,
                ..
            } = inst
            {
                // The stride operand should no longer be `one` (IrId(3))
                if *left == iv && *right != one {
                    stride_updated = true;
                }
            }
        }

        assert!(found_vf_const, "Should insert Const(VF=4)");
        assert!(stride_updated, "Should update stride from 1 to VF");
    }

    #[test]
    fn test_update_loop_bound() {
        let mut function = make_test_function();

        let header = function.cfg.entry_block;
        let body = function.cfg.create_block();
        let exit = function.cfg.create_block();

        let iv = IrId::new(0);
        let cmp_result = IrId::new(1);
        let old_bound = IrId::new(2);
        function.next_reg_id = 3;

        if let Some(hdr) = function.cfg.blocks.get_mut(&header) {
            hdr.instructions = vec![IrInstruction::Cmp {
                dest: cmp_result,
                op: CompareOp::Lt,
                left: iv,
                right: old_bound,
            }];
            hdr.terminator = IrTerminator::CondBranch {
                condition: cmp_result,
                true_target: body,
                false_target: exit,
            };
        }

        let loop_info = make_loop_info(header, body, exit, 16);

        let pass = LoopVectorizationPass::new();
        // 4 vector iterations * VF=4 = adjusted bound of 16
        pass.update_loop_bound(&mut function, &loop_info, 4, 4);

        let hdr = function.cfg.blocks.get(&header).unwrap();
        // Should have a new Const(16) inserted before Cmp, and Cmp.right updated
        let mut found_bound_const = false;
        for inst in &hdr.instructions {
            if let IrInstruction::Const {
                value: IrValue::I32(16),
                ..
            } = inst
            {
                found_bound_const = true;
            }
        }
        assert!(found_bound_const, "Should insert adjusted bound constant");

        // Cmp should now reference the new bound register, not old_bound
        let cmp_inst = hdr
            .instructions
            .iter()
            .find(|i| matches!(i, IrInstruction::Cmp { .. }));
        if let Some(IrInstruction::Cmp { right, .. }) = cmp_inst {
            assert_ne!(*right, old_bound, "Cmp bound should be updated");
        } else {
            panic!("Cmp instruction not found");
        }
    }

    #[test]
    fn test_is_vectorizable_binop() {
        assert!(LoopVectorizationPass::is_vectorizable_binop(BinaryOp::FAdd));
        assert!(LoopVectorizationPass::is_vectorizable_binop(BinaryOp::Mul));
        assert!(!LoopVectorizationPass::is_vectorizable_binop(BinaryOp::Shl));
    }

    #[test]
    fn test_vectorize_instruction_load() {
        let pass = LoopVectorizationPass::new();
        let ptr = IrId::new(0);
        let dest = IrId::new(1);
        let iv = IrId::new(2);

        let accesses = vec![MemoryAccess {
            instruction_id: 0,
            base: ptr,
            stride: 1,
            is_load: true,
            element_type: IrType::F32,
        }];

        let inst = IrInstruction::Load {
            dest,
            ptr,
            ty: IrType::F32,
        };

        let vec_type = VectorType::V4F32;
        let result = pass.vectorize_instruction(&inst, &accesses, &vec_type, iv, 4);

        match result {
            IrInstruction::VectorLoad {
                dest: d, ptr: p, ..
            } => {
                assert_eq!(d, dest);
                assert_eq!(p, ptr);
            }
            other => panic!("Expected VectorLoad, got {:?}", other),
        }
    }

    #[test]
    fn test_epilogue_creation() {
        let mut function = make_test_function();

        let header = function.cfg.entry_block;
        let body = function.cfg.create_block();
        let exit = function.cfg.create_block();
        function.next_reg_id = 10;

        // Header with condbranch
        if let Some(hdr) = function.cfg.blocks.get_mut(&header) {
            hdr.terminator = IrTerminator::CondBranch {
                condition: IrId::new(0),
                true_target: body,
                false_target: exit,
            };
        }

        // Body with a vectorized load
        let ptr = IrId::new(5);
        let loaded = IrId::new(6);
        if let Some(b) = function.cfg.blocks.get_mut(&body) {
            b.instructions = vec![IrInstruction::VectorLoad {
                dest: loaded,
                ptr,
                vec_ty: VectorType::V4F32.to_ir_type(),
            }];
            b.terminator = IrTerminator::Branch { target: header };
        }

        if let Some(e) = function.cfg.blocks.get_mut(&exit) {
            e.terminator = IrTerminator::Return { value: None };
            e.predecessors = vec![header];
        }

        let loop_info = make_loop_info(header, body, exit, 18);

        let pass = LoopVectorizationPass::new();
        let block_count_before = function.cfg.blocks.len();
        pass.create_epilogue_loop(&mut function, &loop_info, 2, &IrType::F32);

        // Should have created a new epilogue block
        assert_eq!(
            function.cfg.blocks.len(),
            block_count_before + 1,
            "Epilogue block should be created"
        );

        // Header's false branch should now point to epilogue, not exit
        let hdr = function.cfg.blocks.get(&header).unwrap();
        if let IrTerminator::CondBranch { false_target, .. } = &hdr.terminator {
            assert_ne!(
                *false_target, exit,
                "Header should branch to epilogue, not exit"
            );
        }
    }
}
