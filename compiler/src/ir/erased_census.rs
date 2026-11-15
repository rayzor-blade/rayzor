//! Census of scalars that cross an erased generic slot without a word
//! conversion (`runtime/src/erased_word.rs`). It changes nothing; it reports
//! each crossing so the representation change has a worklist.
//!
//! A crossing is a register whose type is a scalar (Int, Float, Single, Bool,
//! Null<Int>) meeting a slot typed `i64` or a type variable:
//! - `arg`: a direct call to a function with a body passes one into such a
//!   parameter;
//! - `ret`: such a call's result lands in a scalar register;
//! - `bits`: a bitcast moves a Float between the two;
//! - `closure`: a function value is cast between signatures that disagree on
//!   one parameter or the return in the same way.

use super::{IrFunction, IrId, IrInstruction, IrModule, IrType};
use std::collections::BTreeMap;

fn is_scalar(ty: &IrType) -> bool {
    matches!(
        ty,
        IrType::I32 | IrType::F64 | IrType::F32 | IrType::Bool | IrType::NullInt
    )
}

fn is_word(ty: &IrType) -> bool {
    matches!(ty, IrType::I64 | IrType::TypeVar(_))
}

fn crosses(a: &IrType, b: &IrType) -> bool {
    (is_scalar(a) && is_word(b)) || (is_word(a) && is_scalar(b))
}

/// Register types, constants included: a constant's register has no entry
/// in `register_types`.
fn register_types(function: &IrFunction) -> BTreeMap<IrId, IrType> {
    let mut types = function.register_types.clone();
    for p in &function.signature.parameters {
        types.entry(p.reg).or_insert_with(|| p.ty.clone());
    }
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::Const { dest, value } = inst {
                types
                    .entry(*dest)
                    .or_insert_with(|| super::validation::value_type(value));
            }
        }
    }
    types
}

fn closure_crosses(from: &IrType, to: &IrType) -> bool {
    let (
        IrType::Function {
            params: from_params,
            return_type: from_ret,
            ..
        },
        IrType::Function {
            params: to_params,
            return_type: to_ret,
            ..
        },
    ) = (from, to)
    else {
        return false;
    };
    crosses(from_ret, to_ret)
        || from_params
            .iter()
            .zip(to_params)
            .any(|(a, b)| crosses(a, b))
}

fn function_label(function: &IrFunction) -> &str {
    function.qualified_name.as_deref().unwrap_or(&function.name)
}

/// Print every crossing in `module` to stderr, then a count per kind.
pub fn report(module: &IrModule, label: &str) {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut note = |kind: &'static str, function: &IrFunction, detail: String| {
        *counts.entry(kind).or_default() += 1;
        eprintln!(
            "[erased-census] {kind} in {}: {detail}",
            function_label(function)
        );
    };
    for function in module.functions.values() {
        let types = register_types(function);
        let register_type = |reg: IrId| types.get(&reg);
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::CallDirect {
                        dest,
                        func_id,
                        args,
                        ..
                    } => {
                        let Some(callee) = module.functions.get(func_id) else {
                            continue;
                        };
                        if callee.cfg.blocks.is_empty() {
                            continue;
                        }
                        for (i, (arg, param)) in
                            args.iter().zip(&callee.signature.parameters).enumerate()
                        {
                            if let Some(ty) = register_type(*arg) {
                                if is_scalar(ty) && is_word(&param.ty) {
                                    note(
                                        "arg",
                                        function,
                                        format!(
                                            "{ty} into {} param {i} ({})",
                                            function_label(callee),
                                            param.ty
                                        ),
                                    );
                                }
                            }
                        }
                        let ret = &callee.signature.return_type;
                        if let Some(ty) = dest.and_then(register_type) {
                            if is_word(ret) && is_scalar(ty) {
                                note(
                                    "ret",
                                    function,
                                    format!("{ret} from {} as {ty}", function_label(callee)),
                                );
                            }
                        }
                    }
                    IrInstruction::BitCast { src, ty, .. } => {
                        if let Some(from) = register_type(*src) {
                            let float = |t: &IrType| matches!(t, IrType::F64 | IrType::F32);
                            if (float(from) && is_word(ty)) || (is_word(from) && float(ty)) {
                                note("bits", function, format!("{from} to {ty}"));
                            }
                        }
                    }
                    IrInstruction::Cast { from_ty, to_ty, .. } => {
                        if closure_crosses(from_ty, to_ty) {
                            note("closure", function, format!("{from_ty} to {to_ty}"));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if counts.is_empty() {
        return;
    }
    let summary: Vec<String> = counts.iter().map(|(k, n)| format!("{k}={n}")).collect();
    eprintln!("[erased-census] {label}: {}", summary.join(" "));
}
