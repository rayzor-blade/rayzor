/// Instruction lowering for MIR → Cranelift IR
///
/// This module handles the translation of MIR instructions to Cranelift IR.
/// Based on tested implementation from Zyntax compiler.
use cranelift::prelude::*;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use std::collections::HashMap;

use crate::ir::{BinaryOp, CompareOp, IrId, IrInstruction, IrType, UnaryOp};

use super::CraneliftBackend;

impl CraneliftBackend {
    /// Lower a binary operation to Cranelift IR
    pub(super) fn lower_binary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        op: &BinaryOp,
        ty: &IrType,
        left: IrId,
        right: IrId,
    ) -> Result<Value, String> {
        let lhs = *self.value_map.get(&left).ok_or("Left operand not found")?;
        let rhs = *self
            .value_map
            .get(&right)
            .ok_or("Right operand not found")?;

        // Check if operands have matching types and cast if needed
        // Same logic as lower_binary_op_static to handle i32/i64 type mismatches
        let lhs_ty = builder.func.dfg.value_type(lhs);
        let rhs_ty = builder.func.dfg.value_type(rhs);

        // Convert the MIR type to Cranelift type to get the expected operation type
        let expected_ty = match ty {
            IrType::I32 => types::I32,
            IrType::I64 => types::I64,
            IrType::U32 => types::I32,
            IrType::U64 => types::I64,
            IrType::F32 => types::F32,
            IrType::F64 => types::F64,
            IrType::Bool => types::I32,
            _ => types::I64, // Default to I64 for other types
        };

        // Coerce both operands to a common type - always use the larger of the two
        // to avoid truncating pointers (i64 values from generic functions)
        // Also handle the case where MIR type is float but operands are integers
        let both_operands_are_int = lhs_ty.is_int() && rhs_ty.is_int();
        let larger_operand_ty = if lhs_ty.bits() >= rhs_ty.bits() {
            lhs_ty
        } else {
            rhs_ty
        };

        let operation_ty = if both_operands_are_int && !expected_ty.is_int() {
            // Expected type is not an integer but operands are - use the larger operand type
            // This happens with unresolved generics that become Ptr(Void) -> I64
            larger_operand_ty
        } else if larger_operand_ty.is_int()
            && expected_ty.is_int()
            && larger_operand_ty.bits() > expected_ty.bits()
        {
            // Operand type is larger than expected - use larger to prevent truncation
            larger_operand_ty
        } else if expected_ty.is_int() {
            expected_ty
        } else {
            // For non-integer operations (float), use expected type
            expected_ty
        };

        // Determine if we're doing float operations based on operand types
        let use_float_ops = lhs_ty.is_float() || rhs_ty.is_float() || operation_ty.is_float();

        // Coerce operands to the correct type for the operation
        let lhs = if use_float_ops && lhs_ty.is_int() {
            // Convert integer to float
            let float_ty = if rhs_ty.is_float() {
                rhs_ty
            } else {
                types::F64
            };
            if ty.is_signed() || lhs_ty == types::I32 || lhs_ty == types::I64 {
                builder.ins().fcvt_from_sint(float_ty, lhs)
            } else {
                builder.ins().fcvt_from_uint(float_ty, lhs)
            }
        } else if lhs_ty != operation_ty && lhs_ty.is_int() && operation_ty.is_int() {
            // Extend integer to larger integer
            if ty.is_signed() {
                builder.ins().sextend(operation_ty, lhs)
            } else {
                builder.ins().uextend(operation_ty, lhs)
            }
        } else {
            lhs
        };

        let rhs = if use_float_ops && rhs_ty.is_int() {
            // Convert integer to float
            let float_ty = if lhs_ty.is_float() {
                lhs_ty
            } else {
                types::F64
            };
            if ty.is_signed() || rhs_ty == types::I32 || rhs_ty == types::I64 {
                builder.ins().fcvt_from_sint(float_ty, rhs)
            } else {
                builder.ins().fcvt_from_uint(float_ty, rhs)
            }
        } else if rhs_ty != operation_ty && rhs_ty.is_int() && operation_ty.is_int() {
            // Extend integer to larger integer
            if ty.is_signed() {
                builder.ins().sextend(operation_ty, rhs)
            } else {
                builder.ins().uextend(operation_ty, rhs)
            }
        } else {
            rhs
        };

        let value = match op {
            BinaryOp::Add => {
                if use_float_ops {
                    // Fuse multiply-add: fadd(fmul(a, b), c) → fma(a, b, c)
                    if let Some((a, b)) = Self::try_extract_fmul(builder, lhs) {
                        builder.ins().fma(a, b, rhs)
                    } else if let Some((a, b)) = Self::try_extract_fmul(builder, rhs) {
                        builder.ins().fma(a, b, lhs)
                    } else {
                        builder.ins().fadd(lhs, rhs)
                    }
                } else {
                    builder.ins().iadd(lhs, rhs)
                }
            }
            BinaryOp::Sub => {
                if use_float_ops {
                    // Fuse multiply-subtract: fsub(fmul(a, b), c) → fma(a, b, fneg(c))
                    if let Some((a, b)) = Self::try_extract_fmul(builder, lhs) {
                        let neg_rhs = builder.ins().fneg(rhs);
                        builder.ins().fma(a, b, neg_rhs)
                    } else if let Some((a, b)) = Self::try_extract_fmul(builder, rhs) {
                        let neg_a = builder.ins().fneg(a);
                        builder.ins().fma(neg_a, b, lhs)
                    } else {
                        builder.ins().fsub(lhs, rhs)
                    }
                } else {
                    builder.ins().isub(lhs, rhs)
                }
            }
            BinaryOp::Mul => {
                if use_float_ops {
                    builder.ins().fmul(lhs, rhs)
                } else {
                    builder.ins().imul(lhs, rhs)
                }
            }
            BinaryOp::Div => {
                if use_float_ops {
                    builder.ins().fdiv(lhs, rhs)
                } else if ty.is_signed() {
                    builder.ins().sdiv(lhs, rhs)
                } else {
                    builder.ins().udiv(lhs, rhs)
                }
            }
            BinaryOp::Rem => {
                // Get actual types AFTER coercion (lhs/rhs may have been converted above)
                let actual_lhs_ty = builder.func.dfg.value_type(lhs);
                let actual_rhs_ty = builder.func.dfg.value_type(rhs);

                if actual_lhs_ty.is_float() || actual_rhs_ty.is_float() {
                    // Float modulo: a % b = a - floor(a/b) * b
                    // lhs and rhs are already converted to float by the general coercion above
                    let div = builder.ins().fdiv(lhs, rhs);
                    let floored = builder.ins().floor(div);
                    let mul = builder.ins().fmul(floored, rhs);
                    builder.ins().fsub(lhs, mul)
                } else if ty.is_signed() {
                    builder.ins().srem(lhs, rhs)
                } else {
                    builder.ins().urem(lhs, rhs)
                }
            }
            BinaryOp::And => builder.ins().band(lhs, rhs),
            BinaryOp::Or => builder.ins().bor(lhs, rhs),
            BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
            BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
            BinaryOp::Shr => {
                if ty.is_signed() {
                    builder.ins().sshr(lhs, rhs)
                } else {
                    builder.ins().ushr(lhs, rhs)
                }
            }
            BinaryOp::Ushr => builder.ins().ushr(lhs, rhs),
            // Floating point operations
            BinaryOp::FAdd => builder.ins().fadd(lhs, rhs),
            BinaryOp::FSub => builder.ins().fsub(lhs, rhs),
            BinaryOp::FMul => builder.ins().fmul(lhs, rhs),
            BinaryOp::FDiv => builder.ins().fdiv(lhs, rhs),
            BinaryOp::FRem => {
                // Float modulo: a % b = a - floor(a/b) * b
                // Get actual types AFTER coercion (lhs/rhs may have been converted above)
                let actual_lhs_ty = builder.func.dfg.value_type(lhs);
                let actual_rhs_ty = builder.func.dfg.value_type(rhs);

                // Ensure we have float values - convert if needed
                let float_ty = if actual_lhs_ty.is_float() {
                    actual_lhs_ty
                } else if actual_rhs_ty.is_float() {
                    actual_rhs_ty
                } else {
                    types::F64
                };

                let lhs_f = if actual_lhs_ty.is_int() {
                    builder.ins().fcvt_from_sint(float_ty, lhs)
                } else {
                    lhs
                };

                let rhs_f = if actual_rhs_ty.is_int() {
                    builder.ins().fcvt_from_sint(float_ty, rhs)
                } else {
                    rhs
                };

                let div = builder.ins().fdiv(lhs_f, rhs_f);
                let floored = builder.ins().floor(div);
                let mul = builder.ins().fmul(floored, rhs_f);
                builder.ins().fsub(lhs_f, mul)
            }
        };

        // Convert result back to expected type if there's a type mismatch
        // This handles cases where we did float ops but expected an integer result
        let result_ty = builder.func.dfg.value_type(value);
        let final_value = if result_ty.is_float() && expected_ty.is_int() {
            // Convert float result to integer (truncate towards zero)
            if ty.is_signed() {
                builder.ins().fcvt_to_sint(expected_ty, value)
            } else {
                builder.ins().fcvt_to_uint(expected_ty, value)
            }
        } else if result_ty.is_int() && expected_ty.is_int() && result_ty != expected_ty {
            // Integer size mismatch - extend or truncate
            if result_ty.bits() < expected_ty.bits() {
                if ty.is_signed() {
                    builder.ins().sextend(expected_ty, value)
                } else {
                    builder.ins().uextend(expected_ty, value)
                }
            } else {
                // Truncate (this is rare but can happen)
                builder.ins().ireduce(expected_ty, value)
            }
        } else {
            value
        };

        Ok(final_value)
    }

    /// Lower a comparison operation to Cranelift IR
    pub(super) fn lower_compare_op(
        &mut self,
        builder: &mut FunctionBuilder,
        op: &CompareOp,
        ty: &IrType,
        left: IrId,
        right: IrId,
    ) -> Result<Value, String> {
        let lhs = *self.value_map.get(&left).ok_or("Left operand not found")?;
        let rhs = *self
            .value_map
            .get(&right)
            .ok_or("Right operand not found")?;

        // Floating point comparisons
        if ty.is_float()
            || matches!(
                op,
                CompareOp::FEq
                    | CompareOp::FNe
                    | CompareOp::FLt
                    | CompareOp::FLe
                    | CompareOp::FGt
                    | CompareOp::FGe
                    | CompareOp::FOrd
                    | CompareOp::FUno
            )
        {
            let cc = match op {
                CompareOp::Eq | CompareOp::FEq => FloatCC::Equal,
                CompareOp::Ne | CompareOp::FNe => FloatCC::NotEqual,
                CompareOp::Lt | CompareOp::FLt => FloatCC::LessThan,
                CompareOp::Le | CompareOp::FLe => FloatCC::LessThanOrEqual,
                CompareOp::Gt | CompareOp::FGt => FloatCC::GreaterThan,
                CompareOp::Ge | CompareOp::FGe => FloatCC::GreaterThanOrEqual,
                CompareOp::FOrd => FloatCC::Ordered,
                CompareOp::FUno => FloatCC::Unordered,
                _ => return Err(format!("Invalid float comparison: {:?}", op)),
            };
            let cmp = builder.ins().fcmp(cc, lhs, rhs);
            // Return the i8 boolean result directly - don't extend to i32
            // Bool is represented as i8 in the type system
            Ok(cmp)
        } else {
            // Integer comparisons
            let cc = match op {
                CompareOp::Eq => IntCC::Equal,
                CompareOp::Ne => IntCC::NotEqual,
                CompareOp::Lt => IntCC::SignedLessThan,
                CompareOp::Le => IntCC::SignedLessThanOrEqual,
                CompareOp::Gt => IntCC::SignedGreaterThan,
                CompareOp::Ge => IntCC::SignedGreaterThanOrEqual,
                CompareOp::ULt => IntCC::UnsignedLessThan,
                CompareOp::ULe => IntCC::UnsignedLessThanOrEqual,
                CompareOp::UGt => IntCC::UnsignedGreaterThan,
                CompareOp::UGe => IntCC::UnsignedGreaterThanOrEqual,
                _ => return Err(format!("Invalid int comparison: {:?}", op)),
            };
            let cmp = builder.ins().icmp(cc, lhs, rhs);
            // Return the i8 boolean result directly - don't extend to i32
            // Bool is represented as i8 in the type system
            Ok(cmp)
        }
    }

    /// Lower a unary operation to Cranelift IR
    pub(super) fn lower_unary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        op: &UnaryOp,
        ty: &IrType,
        operand: IrId,
    ) -> Result<Value, String> {
        let val = *self.value_map.get(&operand).ok_or("Operand not found")?;

        let value = match op {
            UnaryOp::Neg => {
                if ty.is_float() {
                    builder.ins().fneg(val)
                } else {
                    builder.ins().ineg(val)
                }
            }
            UnaryOp::Not => {
                if *ty == IrType::Bool {
                    // Logical NOT: compare == 0, producing a proper 0/1 result
                    builder.ins().icmp_imm(IntCC::Equal, val, 0)
                } else {
                    builder.ins().bnot(val)
                }
            }
            UnaryOp::FNeg => builder.ins().fneg(val),
        };

        Ok(value)
    }

    /// Lower a load instruction to Cranelift IR
    pub(super) fn lower_load(
        &mut self,
        builder: &mut FunctionBuilder,
        ty: &IrType,
        ptr: IrId,
    ) -> Result<Value, String> {
        let ptr_val = *self.value_map.get(&ptr).ok_or("Pointer not found")?;
        let cranelift_ty = self.mir_type_to_cranelift(ty)?;

        let flags = MemFlags::new().with_aligned().with_notrap();
        let value = builder.ins().load(cranelift_ty, flags, ptr_val, 0);

        Ok(value)
    }

    /// Lower a store instruction to Cranelift IR
    pub(super) fn lower_store(
        &mut self,
        builder: &mut FunctionBuilder,
        value: IrId,
        ptr: IrId,
    ) -> Result<(), String> {
        let val = *self.value_map.get(&value).ok_or("Value not found")?;
        let ptr_val = *self.value_map.get(&ptr).ok_or("Pointer not found")?;

        let flags = MemFlags::new().with_aligned().with_notrap();
        builder.ins().store(flags, val, ptr_val, 0);

        Ok(())
    }

    /// Lower an alloca instruction to Cranelift IR (using stack slots)
    pub(super) fn lower_alloca(
        &mut self,
        builder: &mut FunctionBuilder,
        ty: &IrType,
        count: Option<u32>,
    ) -> Result<Value, String> {
        let size = type_size(ty)?;
        let alloc_size = if let Some(c) = count { size * c } else { size };

        let slot_data = StackSlotData::new(StackSlotKind::ExplicitSlot, alloc_size, 8); // 8-byte alignment
        let slot = builder.create_sized_stack_slot(slot_data);
        let addr = builder.ins().stack_addr(
            types::I64, // Pointer type
            slot,
            0,
        );

        Ok(addr)
    }

    // =========================================================================
    // Static versions of lowering methods (for use without &mut self borrow)
    // =========================================================================

    /// Lower a binary operation (static version)
    pub(super) fn lower_binary_op_static(
        value_map: &HashMap<IrId, Value>,
        builder: &mut FunctionBuilder,
        op: &BinaryOp,
        ty: &IrType,
        left: IrId,
        right: IrId,
    ) -> Result<Value, String> {
        let lhs = *value_map.get(&left).ok_or_else(|| {
            eprintln!(
                "ERROR: Left operand {:?} not found. Available keys: {:?}",
                left,
                value_map.keys().collect::<Vec<_>>()
            );
            format!("Left operand {:?} not found in value_map", left)
        })?;
        let rhs = *value_map
            .get(&right)
            .ok_or_else(|| format!("Right operand {:?} not found in value_map", right))?;

        // Check if operands have matching types and cast if needed
        // IMPORTANT: Coerce operands to match the expected operation type (ty),
        // not just each other. This is critical for closure environments where
        // captured variables are stored as i64 but operations may expect i32.
        let lhs_ty = builder.func.dfg.value_type(lhs);
        let rhs_ty = builder.func.dfg.value_type(rhs);

        // Convert the MIR type to Cranelift type to get the expected operation type
        let expected_ty = match ty {
            IrType::I32 => types::I32,
            IrType::I64 => types::I64,
            IrType::U32 => types::I32,
            IrType::U64 => types::I64,
            IrType::F32 => types::F32,
            IrType::F64 => types::F64,
            IrType::Bool => types::I32,
            _ => types::I64, // Default to I64 for other types
        };

        // Coerce both operands to a common type for the operation.
        // IMPORTANT: We NEVER truncate from I64 to I32 because the I64 value might be
        // a pointer loaded from a closure environment. Instead, we always extend to
        // the larger type (I64) and perform the operation at that width.
        //
        // The basic strategy is:
        // - If both operands are integers, use the larger of the two operand types
        // - If operands differ, extend the smaller one to match the larger
        // - Never reduce I64 to I32 (would corrupt pointers)
        // - CRITICAL: When expected_ty is float but operands are integers, use integer coercion
        //   (this happens with generic types that resolve to different types)

        // Determine operation type based on actual operand types
        // When both operands are integers but expected is not, use the larger integer type
        let both_operands_are_int = lhs_ty.is_int() && rhs_ty.is_int();
        let larger_operand_ty = if lhs_ty.bits() >= rhs_ty.bits() {
            lhs_ty
        } else {
            rhs_ty
        };

        let operation_ty = if both_operands_are_int && !expected_ty.is_int() {
            // Expected type is not an integer but operands are - use the larger operand type
            // This happens with unresolved generics that become Ptr(Void) -> I64
            larger_operand_ty
        } else if larger_operand_ty.is_int()
            && expected_ty.is_int()
            && larger_operand_ty.bits() > expected_ty.bits()
        {
            // Operand type is larger than expected - use larger to prevent truncation
            larger_operand_ty
        } else if expected_ty.is_int() {
            expected_ty
        } else {
            // For non-integer operations (float), use expected type
            expected_ty
        };

        // Determine if we're doing float operations based on operand types
        let use_float_ops = lhs_ty.is_float() || rhs_ty.is_float() || operation_ty.is_float();

        // Coerce operands to the correct type for the operation
        let lhs = if use_float_ops && lhs_ty.is_int() {
            // Convert integer to float
            let float_ty = if rhs_ty.is_float() {
                rhs_ty
            } else {
                types::F64
            };
            if ty.is_signed() || lhs_ty == types::I32 || lhs_ty == types::I64 {
                builder.ins().fcvt_from_sint(float_ty, lhs)
            } else {
                builder.ins().fcvt_from_uint(float_ty, lhs)
            }
        } else if lhs_ty != operation_ty && lhs_ty.is_int() && operation_ty.is_int() {
            // Extend integer to larger integer
            if ty.is_signed() {
                builder.ins().sextend(operation_ty, lhs)
            } else {
                builder.ins().uextend(operation_ty, lhs)
            }
        } else {
            lhs
        };

        let rhs = if use_float_ops && rhs_ty.is_int() {
            // Convert integer to float
            let float_ty = if lhs_ty.is_float() {
                lhs_ty
            } else {
                types::F64
            };
            if ty.is_signed() || rhs_ty == types::I32 || rhs_ty == types::I64 {
                builder.ins().fcvt_from_sint(float_ty, rhs)
            } else {
                builder.ins().fcvt_from_uint(float_ty, rhs)
            }
        } else if rhs_ty != operation_ty && rhs_ty.is_int() && operation_ty.is_int() {
            // Extend integer to larger integer
            if ty.is_signed() {
                builder.ins().sextend(operation_ty, rhs)
            } else {
                builder.ins().uextend(operation_ty, rhs)
            }
        } else {
            rhs
        };

        let value = match op {
            BinaryOp::Add => {
                if use_float_ops {
                    // Fuse multiply-add: fadd(fmul(a, b), c) → fma(a, b, c)
                    if let Some((a, b)) = Self::try_extract_fmul(builder, lhs) {
                        builder.ins().fma(a, b, rhs)
                    } else if let Some((a, b)) = Self::try_extract_fmul(builder, rhs) {
                        builder.ins().fma(a, b, lhs)
                    } else {
                        builder.ins().fadd(lhs, rhs)
                    }
                } else {
                    builder.ins().iadd(lhs, rhs)
                }
            }
            BinaryOp::Sub => {
                if use_float_ops {
                    // Fuse multiply-subtract: fsub(fmul(a, b), c) → fma(a, b, fneg(c))
                    if let Some((a, b)) = Self::try_extract_fmul(builder, lhs) {
                        let neg_rhs = builder.ins().fneg(rhs);
                        builder.ins().fma(a, b, neg_rhs)
                    } else if let Some((a, b)) = Self::try_extract_fmul(builder, rhs) {
                        let neg_a = builder.ins().fneg(a);
                        builder.ins().fma(neg_a, b, lhs)
                    } else {
                        builder.ins().fsub(lhs, rhs)
                    }
                } else {
                    builder.ins().isub(lhs, rhs)
                }
            }
            BinaryOp::Mul => {
                if use_float_ops {
                    builder.ins().fmul(lhs, rhs)
                } else {
                    builder.ins().imul(lhs, rhs)
                }
            }
            BinaryOp::Div => {
                if use_float_ops {
                    builder.ins().fdiv(lhs, rhs)
                } else if ty.is_signed() {
                    builder.ins().sdiv(lhs, rhs)
                } else {
                    builder.ins().udiv(lhs, rhs)
                }
            }
            BinaryOp::Rem => {
                // Get actual types AFTER coercion (lhs/rhs may have been converted above)
                let actual_lhs_ty = builder.func.dfg.value_type(lhs);
                let actual_rhs_ty = builder.func.dfg.value_type(rhs);

                if actual_lhs_ty.is_float() || actual_rhs_ty.is_float() {
                    // Float modulo: a % b = a - floor(a/b) * b
                    // lhs and rhs are already converted to float by the general coercion above
                    let div = builder.ins().fdiv(lhs, rhs);
                    let floored = builder.ins().floor(div);
                    let mul = builder.ins().fmul(floored, rhs);
                    builder.ins().fsub(lhs, mul)
                } else if ty.is_signed() {
                    builder.ins().srem(lhs, rhs)
                } else {
                    builder.ins().urem(lhs, rhs)
                }
            }
            BinaryOp::And => builder.ins().band(lhs, rhs),
            BinaryOp::Or => builder.ins().bor(lhs, rhs),
            BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
            BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
            BinaryOp::Shr => {
                if ty.is_signed() {
                    builder.ins().sshr(lhs, rhs)
                } else {
                    builder.ins().ushr(lhs, rhs)
                }
            }
            BinaryOp::Ushr => builder.ins().ushr(lhs, rhs),
            BinaryOp::FAdd => builder.ins().fadd(lhs, rhs),
            BinaryOp::FSub => builder.ins().fsub(lhs, rhs),
            BinaryOp::FMul => builder.ins().fmul(lhs, rhs),
            BinaryOp::FDiv => builder.ins().fdiv(lhs, rhs),
            BinaryOp::FRem => {
                // Float modulo: a % b = a - floor(a/b) * b
                // Get actual types AFTER coercion (lhs/rhs may have been converted above)
                let actual_lhs_ty = builder.func.dfg.value_type(lhs);
                let actual_rhs_ty = builder.func.dfg.value_type(rhs);

                // Ensure we have float values - convert if needed
                let float_ty = if actual_lhs_ty.is_float() {
                    actual_lhs_ty
                } else if actual_rhs_ty.is_float() {
                    actual_rhs_ty
                } else {
                    types::F64
                };

                let lhs_f = if actual_lhs_ty.is_int() {
                    builder.ins().fcvt_from_sint(float_ty, lhs)
                } else {
                    lhs
                };

                let rhs_f = if actual_rhs_ty.is_int() {
                    builder.ins().fcvt_from_sint(float_ty, rhs)
                } else {
                    rhs
                };

                let div = builder.ins().fdiv(lhs_f, rhs_f);
                let floored = builder.ins().floor(div);
                let mul = builder.ins().fmul(floored, rhs_f);
                builder.ins().fsub(lhs_f, mul)
            }
        };

        // Convert result back to expected type if there's a type mismatch
        // This handles cases where we did float ops but expected an integer result
        let result_ty = builder.func.dfg.value_type(value);
        let final_value = if result_ty.is_float() && expected_ty.is_int() {
            // Convert float result to integer (truncate towards zero)
            if ty.is_signed() {
                builder.ins().fcvt_to_sint(expected_ty, value)
            } else {
                builder.ins().fcvt_to_uint(expected_ty, value)
            }
        } else if result_ty.is_int() && expected_ty.is_int() && result_ty != expected_ty {
            // Integer size mismatch - extend or truncate
            if result_ty.bits() < expected_ty.bits() {
                if ty.is_signed() {
                    builder.ins().sextend(expected_ty, value)
                } else {
                    builder.ins().uextend(expected_ty, value)
                }
            } else {
                // Truncate (this is rare but can happen)
                builder.ins().ireduce(expected_ty, value)
            }
        } else {
            value
        };

        Ok(final_value)
    }

    /// Lower a comparison operation (static version)
    pub(super) fn lower_compare_op_static(
        value_map: &HashMap<IrId, Value>,
        builder: &mut FunctionBuilder,
        op: &CompareOp,
        ty: &IrType,
        left: IrId,
        right: IrId,
    ) -> Result<Value, String> {
        let lhs = *value_map.get(&left).ok_or_else(|| {
            eprintln!(
                "ERROR: Left operand {:?} not found. Available keys: {:?}",
                left,
                value_map.keys().collect::<Vec<_>>()
            );
            format!("Left operand {:?} not found in value_map", left)
        })?;
        let rhs = *value_map
            .get(&right)
            .ok_or_else(|| format!("Right operand {:?} not found in value_map", right))?;

        // Get actual Cranelift value types
        let lhs_cl_ty = builder.func.dfg.value_type(lhs);
        let rhs_cl_ty = builder.func.dfg.value_type(rhs);

        // Use actual Cranelift types to determine comparison mode (not MIR type)
        // This handles cases where intrinsics return different types than MIR expects
        let use_float_cmp = lhs_cl_ty.is_float() && rhs_cl_ty.is_float();

        // Float comparison operators need float values - convert if needed
        let is_float_op = matches!(
            op,
            CompareOp::FEq
                | CompareOp::FNe
                | CompareOp::FLt
                | CompareOp::FLe
                | CompareOp::FGt
                | CompareOp::FGe
                | CompareOp::FOrd
                | CompareOp::FUno
        );

        if use_float_cmp || (is_float_op && (lhs_cl_ty.is_float() || rhs_cl_ty.is_float())) {
            // Ensure both operands are floats (convert ints to f64 if needed)
            let (float_lhs, float_rhs) = if lhs_cl_ty.is_float() && rhs_cl_ty.is_float() {
                (lhs, rhs)
            } else if lhs_cl_ty.is_float() && rhs_cl_ty.is_int() {
                // Convert rhs int to float
                let rhs_float = builder.ins().fcvt_from_sint(lhs_cl_ty, rhs);
                (lhs, rhs_float)
            } else if lhs_cl_ty.is_int() && rhs_cl_ty.is_float() {
                // Convert lhs int to float
                let lhs_float = builder.ins().fcvt_from_sint(rhs_cl_ty, lhs);
                (lhs_float, rhs)
            } else {
                // Both are ints but MIR wants float comparison - convert both to f64
                use cranelift_codegen::ir::types;
                let lhs_float = builder.ins().fcvt_from_sint(types::F64, lhs);
                let rhs_float = builder.ins().fcvt_from_sint(types::F64, rhs);
                (lhs_float, rhs_float)
            };

            let cc = match op {
                CompareOp::Eq | CompareOp::FEq => FloatCC::Equal,
                CompareOp::Ne | CompareOp::FNe => FloatCC::NotEqual,
                CompareOp::Lt | CompareOp::FLt => FloatCC::LessThan,
                CompareOp::Le | CompareOp::FLe => FloatCC::LessThanOrEqual,
                CompareOp::Gt | CompareOp::FGt => FloatCC::GreaterThan,
                CompareOp::Ge | CompareOp::FGe => FloatCC::GreaterThanOrEqual,
                CompareOp::FOrd => FloatCC::Ordered,
                CompareOp::FUno => FloatCC::Unordered,
                _ => return Err(format!("Invalid float comparison: {:?}", op)),
            };
            let cmp = builder.ins().fcmp(cc, float_lhs, float_rhs);
            // Return the i8 boolean result directly - don't extend to i32
            // Bool is represented as i8 in the type system
            Ok(cmp)
        } else {
            // Integer comparison path
            // First, handle mixed float/int types by converting floats to ints
            use cranelift_codegen::ir::types;
            let (int_lhs, int_rhs) = if lhs_cl_ty.is_float() || rhs_cl_ty.is_float() {
                let mut convert_to_int =
                    |val: Value, val_ty: cranelift_codegen::ir::Type| -> Value {
                        if val_ty.is_float() {
                            // Convert float to i64 (truncating toward zero)
                            builder.ins().fcvt_to_sint_sat(types::I64, val)
                        } else {
                            val
                        }
                    };
                (
                    convert_to_int(lhs, lhs_cl_ty),
                    convert_to_int(rhs, rhs_cl_ty),
                )
            } else {
                (lhs, rhs)
            };

            // Get the types after conversion
            let lhs_ty = builder.func.dfg.value_type(int_lhs);
            let rhs_ty = builder.func.dfg.value_type(int_rhs);

            // If types don't match, extend the smaller one to match the larger
            let (final_lhs, final_rhs) = if lhs_ty != rhs_ty && lhs_ty.is_int() && rhs_ty.is_int() {
                let lhs_bits = lhs_ty.bits();
                let rhs_bits = rhs_ty.bits();

                if lhs_bits > rhs_bits {
                    // Extend rhs to match lhs
                    let extended_rhs = if ty.is_signed() {
                        builder.ins().sextend(lhs_ty, int_rhs)
                    } else {
                        builder.ins().uextend(lhs_ty, int_rhs)
                    };
                    (int_lhs, extended_rhs)
                } else {
                    // Extend lhs to match rhs
                    let extended_lhs = if ty.is_signed() {
                        builder.ins().sextend(rhs_ty, int_lhs)
                    } else {
                        builder.ins().uextend(rhs_ty, int_lhs)
                    };
                    (extended_lhs, int_rhs)
                }
            } else {
                (int_lhs, int_rhs)
            };

            let cc = match op {
                CompareOp::Eq => IntCC::Equal,
                CompareOp::Ne => IntCC::NotEqual,
                CompareOp::Lt => IntCC::SignedLessThan,
                CompareOp::Le => IntCC::SignedLessThanOrEqual,
                CompareOp::Gt => IntCC::SignedGreaterThan,
                CompareOp::Ge => IntCC::SignedGreaterThanOrEqual,
                CompareOp::ULt => IntCC::UnsignedLessThan,
                CompareOp::ULe => IntCC::UnsignedLessThanOrEqual,
                CompareOp::UGt => IntCC::UnsignedGreaterThan,
                CompareOp::UGe => IntCC::UnsignedGreaterThanOrEqual,
                _ => return Err(format!("Invalid int comparison: {:?}", op)),
            };
            let cmp = builder.ins().icmp(cc, final_lhs, final_rhs);
            // Return the i8 boolean result directly - don't extend to i32
            // Bool is represented as i8 in the type system
            Ok(cmp)
        }
    }

    /// Lower a unary operation (static version)
    pub(super) fn lower_unary_op_static(
        value_map: &HashMap<IrId, Value>,
        builder: &mut FunctionBuilder,
        op: &UnaryOp,
        ty: &IrType,
        operand: IrId,
    ) -> Result<Value, String> {
        let val = *value_map.get(&operand).ok_or("Operand not found")?;

        let value = match op {
            UnaryOp::Neg => {
                if ty.is_float() {
                    builder.ins().fneg(val)
                } else {
                    builder.ins().ineg(val)
                }
            }
            UnaryOp::Not => {
                if *ty == IrType::Bool {
                    // Logical NOT: compare == 0, producing a proper 0/1 result
                    builder.ins().icmp_imm(IntCC::Equal, val, 0)
                } else {
                    builder.ins().bnot(val)
                }
            }
            UnaryOp::FNeg => builder.ins().fneg(val),
        };

        Ok(value)
    }

    /// Lower a load operation (static version)
    pub(super) fn lower_load_static(
        value_map: &HashMap<IrId, Value>,
        builder: &mut FunctionBuilder,
        ty: &IrType,
        ptr: IrId,
    ) -> Result<Value, String> {
        let ptr_val = *value_map.get(&ptr).ok_or("Pointer not found")?;
        let cranelift_ty = Self::mir_type_to_cranelift_static(ty)?;
        let flags = MemFlags::new().with_aligned().with_notrap();
        let value = builder.ins().load(cranelift_ty, flags, ptr_val, 0);
        Ok(value)
    }

    /// Lower a store operation (static version)
    pub(super) fn lower_store_static(
        value_map: &HashMap<IrId, Value>,
        builder: &mut FunctionBuilder,
        ptr: IrId,
        value: IrId,
    ) -> Result<(), String> {
        let val = *value_map.get(&value).ok_or("Value not found")?;
        let ptr_val = *value_map.get(&ptr).ok_or("Pointer not found")?;
        let flags = MemFlags::new().with_aligned().with_notrap();
        builder.ins().store(flags, val, ptr_val, 0);
        Ok(())
    }

    /// Lower an alloca operation (static version)
    pub(super) fn lower_alloca_static(
        builder: &mut FunctionBuilder,
        ty: &IrType,
        count: Option<u32>,
    ) -> Result<Value, String> {
        let size = type_size(ty)?;
        let alloc_size = if let Some(c) = count {
            size * c
        } else {
            // WORKAROUND: For complex types (Any, Ptr, etc.) that might be dynamic arrays,
            // allocate extra space to avoid stack corruption.
            // Arrays should really be heap-allocated, but for now we allocate enough
            // stack space for reasonable array sizes (up to 16 elements).
            if matches!(ty, IrType::Any | IrType::Ptr(_) | IrType::Ref(_)) {
                size * 16 // Allocate space for up to 16 pointers/elements
            } else {
                size
            }
        };

        let slot_data = StackSlotData::new(StackSlotKind::ExplicitSlot, alloc_size, 8);
        let slot = builder.create_sized_stack_slot(slot_data);
        let addr = builder.ins().stack_addr(types::I64, slot, 0);
        Ok(addr)
    }

    /// Check if a Cranelift value was produced by an fmul instruction in the same block.
    /// Returns the two operands if so, enabling FMA fusion.
    ///
    /// Only fuses when fmul is in the same block as the current insertion point.
    /// Cross-block fusion would change FP semantics for values that originally went
    /// through memory (store/load), since SRA + CopyProp can expose fmul results
    /// across block boundaries that were previously hidden by memory operations.
    fn try_extract_fmul(builder: &FunctionBuilder, value: Value) -> Option<(Value, Value)> {
        if std::env::var("RAYZOR_NO_FMA").is_ok() {
            return None;
        }
        use cranelift_codegen::ir::{InstructionData, Opcode, ValueDef};
        let current_block = builder.current_block()?;
        match builder.func.dfg.value_def(value) {
            ValueDef::Result(inst, 0) => {
                // Only fuse if fmul is in the same block as the fadd/fsub
                let fmul_block = builder.func.layout.inst_block(inst)?;
                if fmul_block != current_block {
                    return None;
                }
                if let InstructionData::Binary { opcode, args } = builder.func.dfg.insts[inst] {
                    if opcode == Opcode::Fmul {
                        return Some((args[0], args[1]));
                    }
                }
                None
            }
            _ => None,
        }
    }
}

/// Calculate type size in bytes
fn type_size(ty: &IrType) -> Result<u32, String> {
    Ok(match ty {
        IrType::I8 | IrType::U8 | IrType::Bool => 1,
        IrType::I16 | IrType::U16 => 2,
        IrType::I32 | IrType::U32 | IrType::F32 => 4,
        IrType::I64 | IrType::U64 | IrType::F64 => 8,
        IrType::Ptr(_) | IrType::Ref(_) | IrType::Any | IrType::Function { .. } => 8,
        _ => 8, // Default to pointer size for complex types
    })
}

// Helper trait to check type properties
pub trait TypeProperties {
    fn is_float(&self) -> bool;
    fn is_signed(&self) -> bool;
}

impl TypeProperties for IrType {
    fn is_float(&self) -> bool {
        matches!(self, IrType::F32 | IrType::F64)
    }

    fn is_signed(&self) -> bool {
        matches!(self, IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64)
    }
}
