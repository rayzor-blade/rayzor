//! `Std.int(a / c)` for an Int `a` and an Int constant `c` is integer
//! division: Haxe's `/` divides in Float, and for 32-bit operands the double
//! quotient is never close enough to an integer it does not reach for the
//! truncation to land on the wrong side. Integer division by a constant
//! lowers to a multiply, where the Float form converts, divides and converts
//! back. `c` is never 0 or -1, the two divisors integer division traps on.

use super::optimization::{OptimizationPass, OptimizationResult};
use super::{BinaryOp, IrFunction, IrId, IrInstruction, IrModule, IrType, IrValue};
use std::collections::BTreeMap;

pub struct IntQuotientPass;

impl IntQuotientPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for IntQuotientPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for IntQuotientPass {
    fn name(&self) -> &'static str {
        "int-quotient"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();
        let std_int: Vec<_> = module
            .extern_functions
            .values()
            .filter(|f| f.name == "haxe_std_int")
            .map(|f| f.id)
            .chain(
                module
                    .functions
                    .values()
                    .filter(|f| f.cfg.blocks.is_empty() && f.name == "haxe_std_int")
                    .map(|f| f.id),
            )
            .collect();
        if std_int.is_empty() {
            return result;
        }
        for function in module.functions.values_mut() {
            let rewritten = rewrite(function, &std_int);
            if rewritten > 0 {
                result.modified = true;
                *result.stats.entry("int_quotients".to_string()).or_insert(0) += rewritten;
            }
        }
        result
    }
}

/// The Int a Float operand was converted from, or the Int constant it is.
enum IntSource {
    Value(IrId),
    Constant(i32),
}

fn rewrite(function: &mut IrFunction, std_int: &[super::IrFunctionId]) -> usize {
    let mut defs: BTreeMap<IrId, IrInstruction> = BTreeMap::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() {
                defs.insert(dest, inst.clone());
            }
        }
    }
    let int_const = |reg: &IrId| match defs.get(reg) {
        Some(IrInstruction::Const {
            value: IrValue::I32(v),
            ..
        }) => Some(*v),
        _ => None,
    };
    let source = |reg: &IrId| match defs.get(reg)? {
        IrInstruction::Cast {
            src,
            from_ty: IrType::I32,
            to_ty: IrType::F64,
            ..
        } => Some(match int_const(src) {
            Some(c) => IntSource::Constant(c),
            None => IntSource::Value(*src),
        }),
        IrInstruction::Const {
            value: IrValue::F64(f),
            ..
        } if f.fract() == 0.0 && f.abs() < 2147483648.0 => Some(IntSource::Constant(*f as i32)),
        _ => None,
    };

    // (block, index, call dest, dividend, divisor)
    let mut sites = Vec::new();
    for (block_id, block) in &function.cfg.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            let IrInstruction::CallDirect {
                func_id,
                args,
                dest: Some(dest),
                ..
            } = inst
            else {
                continue;
            };
            if !std_int.contains(func_id) || args.len() != 1 {
                continue;
            }
            let Some(IrInstruction::BinOp {
                op: BinaryOp::Div | BinaryOp::FDiv,
                left,
                right,
                ..
            }) = defs.get(&args[0])
            else {
                continue;
            };
            let (Some(IntSource::Value(dividend)), Some(IntSource::Constant(divisor))) =
                (source(left), source(right))
            else {
                continue;
            };
            if divisor == 0 || divisor == -1 {
                continue;
            }
            sites.push((*block_id, idx, *dest, dividend, divisor));
        }
    }

    // Highest index first, so earlier sites in a block keep their positions.
    sites.sort_by_key(|s| std::cmp::Reverse((s.0, s.1)));
    for &(block_id, idx, dest, dividend, divisor) in &sites {
        let constant = function.alloc_reg();
        let quotient = function.alloc_reg();
        function.register_types.insert(constant, IrType::I32);
        function.register_types.insert(quotient, IrType::I32);
        let to_ty = function
            .register_types
            .get(&dest)
            .cloned()
            .unwrap_or(IrType::I64);
        let replacement = [
            IrInstruction::Const {
                dest: constant,
                value: IrValue::I32(divisor),
            },
            IrInstruction::BinOp {
                dest: quotient,
                op: BinaryOp::Div,
                left: dividend,
                right: constant,
            },
            if to_ty == IrType::I32 {
                IrInstruction::Copy {
                    dest,
                    src: quotient,
                }
            } else {
                IrInstruction::Cast {
                    dest,
                    src: quotient,
                    from_ty: IrType::I32,
                    to_ty,
                }
            },
        ];
        let block = function.cfg.blocks.get_mut(&block_id).unwrap();
        block.instructions.splice(idx..=idx, replacement);
    }
    sites.len()
}
