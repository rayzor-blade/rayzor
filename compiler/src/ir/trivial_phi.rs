//! Trivial phi elimination: a phi whose inputs are all one value, or the phi
//! itself, is that value. Lowering leaves many, carrying a loop-invariant
//! value through every block of a loop, and each one hides the invariance
//! from passes that judge an operand by where it is defined.
//!
//! Replacing the phi by the value is sound: every edge into the phi's block
//! carries the value or the phi, so the value's definition dominates the
//! block and therefore every use of the phi.

use super::optimization::{InstructionExt, OptimizationPass, OptimizationResult};
use super::{IrFunction, IrId, IrModule, IrTerminator};
use std::collections::BTreeMap;

pub struct TrivialPhiEliminationPass;

impl TrivialPhiEliminationPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TrivialPhiEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for TrivialPhiEliminationPass {
    fn name(&self) -> &'static str {
        "trivial-phi-elimination"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();
        for function in module.functions.values_mut() {
            let removed = eliminate(function);
            if removed > 0 {
                result.modified = true;
                *result
                    .stats
                    .entry("trivial_phis_removed".to_string())
                    .or_insert(0) += removed;
            }
        }
        result
    }
}

fn eliminate(function: &mut IrFunction) -> usize {
    let alias = super::loop_unswitch::renamings(function);
    let phi_dests: Vec<IrId> = function
        .cfg
        .blocks
        .values()
        .flat_map(|b| b.phi_nodes.iter().map(|p| p.dest))
        .collect();
    let replace: BTreeMap<IrId, IrId> = phi_dests
        .into_iter()
        .filter_map(|dest| {
            let value = super::loop_unswitch::canonical(&alias, dest);
            (value != dest).then_some((dest, value))
        })
        .collect();
    if replace.is_empty() {
        return 0;
    }
    let get = |reg: &mut IrId| {
        if let Some(value) = replace.get(reg) {
            *reg = *value;
        }
    };
    for block in function.cfg.blocks.values_mut() {
        block
            .phi_nodes
            .retain(|phi| !replace.contains_key(&phi.dest));
        for phi in &mut block.phi_nodes {
            for (_, value) in &mut phi.incoming {
                get(value);
            }
        }
        for inst in &mut block.instructions {
            inst.replace_uses(&replace);
        }
        match &mut block.terminator {
            IrTerminator::CondBranch { condition, .. } => get(condition),
            IrTerminator::Switch { value, .. } => get(value),
            IrTerminator::Return { value: Some(v) } => get(v),
            IrTerminator::NoReturn { call } => get(call),
            _ => {}
        }
    }
    replace.len()
}
