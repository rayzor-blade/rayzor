//! Census of scalars that cross an erased generic slot without a word
//! conversion (`runtime/src/erased_word.rs`). It changes nothing; it reports
//! each crossing so the representation change has a worklist.
//!
//! A crossing is a register whose type is a scalar (Int, Float, Single, Bool)
//! meeting a slot typed `i64` or a type variable. A `Null<Int>` is not one: a
//! slot bound to it holds the raw value.
//! - `arg`: a direct call to a function with a body passes one into a
//!   parameter declared at a type parameter;
//! - `ret`: such a call's result lands in a scalar register;
//! - `bits`: a bitcast moves a Float between the two, other than the bits
//!   a word encode reads, bits a native call takes or returns, a round trip
//!   back to the same type, and a closure slot entry's native slot ABI;
//! - `closure`: an indirect call passes one into such a parameter or reads
//!   one from such a result. A closure keeps its target's native signature,
//!   so a cast between function types only relabels it; a call through an
//!   erased type reaches the target through a view, with words.

use super::{IrFunction, IrId, IrInstruction, IrModule, IrType};
use std::collections::BTreeMap;

fn is_scalar(ty: &IrType) -> bool {
    matches!(ty, IrType::I32 | IrType::F64 | IrType::F32 | IrType::Bool)
}

fn is_word(ty: &IrType) -> bool {
    matches!(ty, IrType::I64 | IrType::TypeVar(_))
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

fn has_body(module: &IrModule, id: super::IrFunctionId) -> bool {
    module
        .functions
        .get(&id)
        .is_some_and(|f| !f.cfg.blocks.is_empty())
}

/// A closure's slot-shaped entry, which the runtime's array helpers call
/// with native slot bits (`ensure_closure_slot_entry`).
fn is_slot_entry(function: &IrFunction) -> bool {
    function.name.starts_with("__closure_slot_entry_")
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
        // A word encode reads a Float's bits after testing it for NaN.
        let nan_tested: std::collections::BTreeSet<IrId> = function
            .cfg
            .blocks
            .values()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| match inst {
                IrInstruction::Cmp {
                    op: super::CompareOp::FUno,
                    left,
                    right,
                    ..
                } if left == right => Some(*left),
                _ => None,
            })
            .collect();
        // Bits passed to or returned by a native function travel with the
        // native call's own tag; they never sit in an erased slot.
        let native_bits: std::collections::BTreeSet<IrId> = function
            .cfg
            .blocks
            .values()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| match inst {
                IrInstruction::CallDirect {
                    dest,
                    func_id,
                    args,
                    ..
                } if !has_body(module, *func_id) => Some(args.iter().copied().chain(*dest)),
                _ => None,
            })
            .flatten()
            .collect();
        // The middle register of a bitcast round trip, Float to i64 and back.
        let bitcasts: BTreeMap<IrId, IrId> = function
            .cfg
            .blocks
            .values()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| match inst {
                IrInstruction::BitCast { dest, src, .. } => Some((*dest, *src)),
                _ => None,
            })
            .collect();
        let round_trips: std::collections::BTreeSet<IrId> = bitcasts
            .iter()
            .filter_map(|(dest, mid)| {
                let origin = bitcasts.get(mid)?;
                (register_type(*origin) == register_type(*dest)).then_some(*mid)
            })
            .collect();
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
                            let erased = matches!(param.ty, IrType::TypeVar(_))
                                || callee.erased_params.contains(&i);
                            if let Some(ty) = register_type(*arg) {
                                if is_scalar(ty) && erased {
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
                    IrInstruction::BitCast { dest, src, ty }
                        if !nan_tested.contains(src)
                            && !native_bits.contains(src)
                            && !native_bits.contains(dest)
                            && !round_trips.contains(src)
                            && !round_trips.contains(dest)
                            && !is_slot_entry(function) =>
                    {
                        if let Some(from) = register_type(*src) {
                            let float = |t: &IrType| matches!(t, IrType::F64 | IrType::F32);
                            if (float(from) && is_word(ty)) || (is_word(from) && float(ty)) {
                                note("bits", function, format!("{from} to {ty}"));
                            }
                        }
                    }
                    IrInstruction::CallIndirect {
                        dest,
                        args,
                        signature:
                            IrType::Function {
                                params,
                                return_type,
                                ..
                            },
                        ..
                    } => {
                        for (i, (arg, param)) in args.iter().zip(params).enumerate() {
                            if let Some(ty) = register_type(*arg) {
                                if is_scalar(ty) && is_word(param) {
                                    note(
                                        "closure",
                                        function,
                                        format!("{ty} into param {i} ({param})"),
                                    );
                                }
                            }
                        }
                        if let Some(ty) = dest.and_then(register_type) {
                            if is_word(return_type) && is_scalar(ty) {
                                note("closure", function, format!("{return_type} result as {ty}"));
                            }
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
