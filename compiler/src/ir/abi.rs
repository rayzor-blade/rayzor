//! Hidden parameters, stated once for every backend.
//!
//! A Haxe function can carry parameters its source never mentions: a struct
//! return buffer and a closure environment. Which ones it carries has to be one
//! answer, because a function compiled by one backend is called by code from
//! another -- a function promoted to LLVM is entered from Cranelift code, and a
//! resumed loop returns into it. Two backends deciding this separately means
//! arguments land in the wrong registers, which is a crash with no bad line to
//! point at.

use std::collections::BTreeSet;

use crate::ir::{CallingConvention, IrFunction, IrFunctionId, IrInstruction, IrModule};

/// Functions that can be reached other than by name.
///
/// A closure or vtable entry is called through a pointer, and the caller knows
/// only a signature, so these keep the environment parameter whether or not
/// their own body reads it. `__vtable_init__` names its entries with direct
/// calls, which is how they get into a vtable, so its call targets count too.
pub fn collect_indirect_targets(module: &IrModule, into: &mut BTreeSet<IrFunctionId>) {
    for function in module.functions.values() {
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::FunctionRef { func_id, .. }
                    | IrInstruction::MakeClosure { func_id, .. } => {
                        into.insert(*func_id);
                    }
                    IrInstruction::CallDirect { func_id, .. }
                        if function.name == "__vtable_init__" =>
                    {
                        into.insert(*func_id);
                    }
                    _ => {}
                }
            }
        }
    }
}

/// The same, over every module that will be compiled together.
///
/// Scanning module by module as each is declared gets a different answer
/// depending on the order they arrive in: a function declared in the first
/// module is only learned to be a closure target when the third is scanned, by
/// which time its signature is fixed. Every backend seeds from the whole set
/// before declaring anything.
pub fn indirect_targets(modules: &[IrModule]) -> BTreeSet<IrFunctionId> {
    let mut targets = BTreeSet::new();
    for module in modules {
        collect_indirect_targets(module, &mut targets);
    }
    targets
}

/// Whether this function is entered from outside the program's own code.
fn is_entry(function: &IrFunction) -> bool {
    function.name == "main"
        || function.name == "__vtable_init__"
        || function.name == "__init__"
        || function.name.starts_with("Main")
}

/// Whether this function is eligible for a hidden environment at all.
///
/// An extern follows the C ABI, a C-convention function is a wrapper around
/// one, and a function whose first parameter is already named `env` is a lambda
/// carrying its own.
pub fn carries_hidden_env(function: &IrFunction) -> bool {
    let is_extern = function.cfg.blocks.is_empty();
    let already_has_env = function
        .signature
        .parameters
        .first()
        .is_some_and(|p| p.name == "env");
    !is_extern && !already_has_env && function.signature.calling_convention != CallingConvention::C
}

/// Whether a hidden environment parameter leads this function's parameters.
///
/// `indirect` is the set from [`indirect_targets`], covering every module in
/// the compile. A function that already names its first parameter `env` is a
/// lambda and carries its own; an extern or C-convention function follows the C
/// ABI and carries nothing.
pub fn needs_env_param(
    function: &IrFunction,
    func_id: IrFunctionId,
    indirect: &BTreeSet<IrFunctionId>,
) -> bool {
    carries_hidden_env(function) && (indirect.contains(&func_id) || is_entry(function))
}
