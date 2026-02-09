//! Function Inlining for MIR Optimization
//!
//! This module provides function inlining infrastructure including:
//! - Call graph construction and analysis
//! - Inlining cost model and heuristics
//! - Function body cloning and integration

use super::loop_analysis::{DominatorTree, LoopNestInfo};
use super::optimization::{InstructionExt, OptimizationPass, OptimizationResult};
use super::{
    IrBasicBlock, IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction, IrModule, IrPhiNode,
    IrTerminator, IrType,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

/// A call site in the program.
#[derive(Debug, Clone)]
pub struct CallSite {
    /// Function containing the call
    pub caller: IrFunctionId,
    /// Function being called
    pub callee: IrFunctionId,
    /// Block containing the call
    pub block: IrBlockId,
    /// Index of the call instruction in the block
    pub instruction_index: usize,
    /// Loop nesting depth at call site (higher = hotter)
    pub loop_depth: usize,
    /// Arguments passed to the call
    pub args: Vec<IrId>,
    /// Destination register for call result (if any)
    pub dest: Option<IrId>,
}

/// Call graph for the module.
#[derive(Debug, Clone)]
pub struct CallGraph {
    /// All call sites in the module
    pub call_sites: Vec<CallSite>,
    /// Map from callee to call sites calling it (BTreeMap for deterministic iteration)
    pub callers: BTreeMap<IrFunctionId, Vec<usize>>,
    /// Map from caller to call sites it contains (BTreeMap for deterministic iteration)
    pub callees: BTreeMap<IrFunctionId, Vec<usize>>,
    /// Functions that are recursive (call themselves directly or indirectly)
    pub recursive_functions: BTreeSet<IrFunctionId>,
}

impl CallGraph {
    /// Build call graph from a module.
    pub fn build(module: &IrModule) -> Self {
        let mut call_sites = Vec::new();
        let mut callers: BTreeMap<IrFunctionId, Vec<usize>> = BTreeMap::new();
        let mut callees: BTreeMap<IrFunctionId, Vec<usize>> = BTreeMap::new();

        for (&func_id, function) in &module.functions {
            // Compute loop info for this function to get call site loop depths
            let domtree = DominatorTree::compute(function);
            let loop_info = LoopNestInfo::analyze(function, &domtree);

            for (&block_id, block) in &function.cfg.blocks {
                let loop_depth = loop_info.loop_depth(block_id);

                for (idx, inst) in block.instructions.iter().enumerate() {
                    if let IrInstruction::CallDirect {
                        dest,
                        func_id: callee_id,
                        args,
                        ..
                    } = inst
                    {
                        let call_site = CallSite {
                            caller: func_id,
                            callee: *callee_id,
                            block: block_id,
                            instruction_index: idx,
                            loop_depth,
                            args: args.clone(),
                            dest: *dest,
                        };

                        let site_idx = call_sites.len();
                        call_sites.push(call_site);

                        callers.entry(*callee_id).or_default().push(site_idx);
                        callees.entry(func_id).or_default().push(site_idx);
                    }
                }
            }
        }

        // Find recursive functions using SCC (simplified: just check direct recursion for now)
        let mut recursive_functions = BTreeSet::new();
        for site in &call_sites {
            if site.caller == site.callee {
                recursive_functions.insert(site.caller);
            }
        }

        // Also check for indirect recursion via reachability
        for &func_id in module.functions.keys() {
            if Self::can_reach(
                &callees,
                &call_sites,
                func_id,
                func_id,
                &mut BTreeSet::new(),
            ) {
                recursive_functions.insert(func_id);
            }
        }

        Self {
            call_sites,
            callers,
            callees,
            recursive_functions,
        }
    }

    /// Check if `from` can reach `target` via calls.
    fn can_reach(
        callees: &BTreeMap<IrFunctionId, Vec<usize>>,
        call_sites: &[CallSite],
        from: IrFunctionId,
        target: IrFunctionId,
        visited: &mut BTreeSet<IrFunctionId>,
    ) -> bool {
        if !visited.insert(from) {
            return false;
        }

        if let Some(sites) = callees.get(&from) {
            for &site_idx in sites {
                let callee = call_sites[site_idx].callee;
                if callee == target {
                    return true;
                }
                if Self::can_reach(callees, call_sites, callee, target, visited) {
                    return true;
                }
            }
        }

        false
    }

    /// Get all call sites for a function.
    pub fn get_call_sites(&self, func_id: IrFunctionId) -> &[usize] {
        self.callees
            .get(&func_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Check if a function is recursive.
    pub fn is_recursive(&self, func_id: IrFunctionId) -> bool {
        self.recursive_functions.contains(&func_id)
    }
}

/// Inlining cost model parameters.
#[derive(Debug, Clone)]
pub struct InliningCostModel {
    /// Maximum instruction count for automatic inlining
    pub max_inline_size: usize,
    /// Bonus multiplier for calls in loops (higher = more likely to inline)
    pub loop_depth_bonus: f64,
    /// Penalty for functions with many basic blocks
    pub block_count_penalty: f64,
    /// Bonus for small functions (leaf functions with few instructions)
    pub small_function_bonus: usize,
    /// Maximum total growth allowed (as percentage of original size)
    pub max_growth_percent: usize,
}

impl Default for InliningCostModel {
    fn default() -> Self {
        Self {
            max_inline_size: 50,      // Inline functions up to 50 instructions
            loop_depth_bonus: 2.0,    // Double threshold for each loop level
            block_count_penalty: 0.9, // Reduce threshold by 10% per extra block
            small_function_bonus: 20, // Extra budget for tiny functions
            max_growth_percent: 200,  // Allow up to 2x code growth
        }
    }
}

impl InliningCostModel {
    /// Calculate the cost of inlining a function at a call site.
    pub fn should_inline(
        &self,
        callee: &IrFunction,
        call_site: &CallSite,
        call_graph: &CallGraph,
    ) -> bool {
        // Never inline recursive functions (for now)
        if call_graph.is_recursive(call_site.callee) {
            return false;
        }

        // Can't inline if entry block doesn't exist (extern/declaration)
        if !callee.cfg.blocks.contains_key(&callee.cfg.entry_block) {
            return false;
        }

        // Always inline functions marked with InlineHint::Always (Haxe `inline` keyword)
        if callee.attributes.inline == super::InlineHint::Always {
            return true;
        }

        // Count instructions and blocks
        let mut inst_count = 0;
        for block in callee.cfg.blocks.values() {
            inst_count += block.instructions.len();
        }
        let block_count = callee.cfg.blocks.len();

        // Calculate adjusted threshold based on call site context
        let mut threshold = self.max_inline_size as f64;

        // Bonus for loops: more likely to inline hot code
        threshold *= self.loop_depth_bonus.powi(call_site.loop_depth as i32);

        // Penalty for complex CFG
        if block_count > 3 {
            threshold *= self.block_count_penalty.powi((block_count - 3) as i32);
        }

        // Bonus for very small functions
        if inst_count <= 5 {
            threshold += self.small_function_bonus as f64;
        }

        inst_count as f64 <= threshold
    }
}

/// Function inlining optimization pass.
pub struct InliningPass {
    /// Cost model for inlining decisions
    cost_model: InliningCostModel,
    /// Maximum iterations of inlining
    max_iterations: usize,
}

impl InliningPass {
    pub fn new() -> Self {
        Self {
            cost_model: InliningCostModel::default(),
            max_iterations: 5,
        }
    }

    pub fn with_cost_model(cost_model: InliningCostModel) -> Self {
        Self {
            cost_model,
            max_iterations: 5,
        }
    }

    /// Inline a specific call site.
    fn inline_call_site(
        module: &mut IrModule,
        call_site: &CallSite,
        next_reg_id: &mut u32,
    ) -> Result<(), String> {
        // Get callee function (clone to avoid borrow issues)
        let callee = module
            .functions
            .get(&call_site.callee)
            .ok_or_else(|| format!("Callee function {:?} not found", call_site.callee))?
            .clone();

        // Get caller function
        let caller = module
            .functions
            .get_mut(&call_site.caller)
            .ok_or_else(|| format!("Caller function {:?} not found", call_site.caller))?;

        // Create register mapping: callee registers -> new registers in caller
        // Use BTreeMap for deterministic iteration order
        let mut reg_map: BTreeMap<IrId, IrId> = BTreeMap::new();

        // Map callee parameters to call arguments
        for (param, arg) in callee.signature.parameters.iter().zip(&call_site.args) {
            reg_map.insert(param.reg, *arg);
        }

        // Allocate new registers for callee's internal values
        for block in callee.cfg.blocks.values() {
            for phi in &block.phi_nodes {
                if !reg_map.contains_key(&phi.dest) {
                    let new_reg = IrId::new(*next_reg_id);
                    *next_reg_id += 1;
                    reg_map.insert(phi.dest, new_reg);
                }
            }
            for inst in &block.instructions {
                if let Some(dest) = inst.dest() {
                    if !reg_map.contains_key(&dest) {
                        let new_reg = IrId::new(*next_reg_id);
                        *next_reg_id += 1;
                        reg_map.insert(dest, new_reg);
                    }
                }
            }
        }

        // Copy register types from callee to caller with remapped IDs
        for (old_reg, new_reg) in &reg_map {
            if let Some(ty) = callee.register_types.get(old_reg) {
                caller.register_types.insert(*new_reg, ty.clone());
            }
        }

        // Create block mapping: callee blocks -> new blocks in caller
        // Use BTreeMap for deterministic iteration order
        let mut block_map: BTreeMap<IrBlockId, IrBlockId> = BTreeMap::new();
        for &block_id in callee.cfg.blocks.keys() {
            let new_block = caller.cfg.create_block();
            block_map.insert(block_id, new_block);
        }

        // Create continuation block for after inlined code
        let continuation_block = caller.cfg.create_block();

        // Collect return block info for phi node in continuation block.
        // When a callee has multiple return paths, we need a phi node to merge
        // the return values from different blocks (instead of invalid multi-def Copy).
        let mut return_phi_incoming: Vec<(IrBlockId, IrId)> = Vec::new();

        // Clone callee blocks into caller with remapped registers and blocks
        for (&old_block_id, old_block) in &callee.cfg.blocks {
            let new_block_id = *block_map
                .get(&old_block_id)
                .ok_or_else(|| format!("Block {:?} not found in block_map", old_block_id))?;

            // Clone and remap phi nodes
            let mut new_phis: Vec<IrPhiNode> = Vec::new();
            for phi in &old_block.phi_nodes {
                let dest = *reg_map
                    .get(&phi.dest)
                    .ok_or_else(|| format!("Phi dest {:?} not found in reg_map", phi.dest))?;

                let mut incoming = Vec::new();
                for (block, val) in &phi.incoming {
                    let new_block = *block_map.get(block).ok_or_else(|| {
                        format!("Phi incoming block {:?} not found in block_map", block)
                    })?;
                    let new_val = *reg_map.get(val).ok_or_else(|| {
                        format!(
                            "Phi incoming value {:?} not found in reg_map (phi dest={:?}, from callee block {:?}). reg_map keys: {:?}",
                            val, phi.dest, block, reg_map.keys().collect::<Vec<_>>()
                        )
                    })?;
                    incoming.push((new_block, new_val));
                }

                new_phis.push(IrPhiNode {
                    dest,
                    incoming,
                    ty: phi.ty.clone(),
                });
            }

            // Clone and remap instructions
            let mut new_instructions: Vec<IrInstruction> = old_block
                .instructions
                .iter()
                .map(|inst| Self::remap_instruction(inst, &reg_map, &block_map))
                .collect();

            // Handle terminator
            let new_terminator = match &old_block.terminator {
                IrTerminator::Return { value } => {
                    // Collect return value for merging at continuation block.
                    if let (Some(_dest), Some(val)) = (call_site.dest, value) {
                        let mapped_val = *reg_map.get(val).unwrap_or(val);
                        return_phi_incoming.push((new_block_id, mapped_val));
                    }
                    IrTerminator::Branch {
                        target: continuation_block,
                    }
                }
                IrTerminator::Branch { target } => {
                    let new_target = *block_map.get(target).ok_or_else(|| {
                        format!("Branch target {:?} not found in block_map", target)
                    })?;
                    IrTerminator::Branch { target: new_target }
                }
                IrTerminator::CondBranch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    let new_true = *block_map.get(true_target).ok_or_else(|| {
                        format!("CondBranch true_target {:?} not found", true_target)
                    })?;
                    let new_false = *block_map.get(false_target).ok_or_else(|| {
                        format!("CondBranch false_target {:?} not found", false_target)
                    })?;
                    IrTerminator::CondBranch {
                        condition: *reg_map.get(condition).unwrap_or(condition),
                        true_target: new_true,
                        false_target: new_false,
                    }
                }
                IrTerminator::Switch {
                    value,
                    cases,
                    default,
                } => {
                    let mut new_cases = Vec::new();
                    for (v, t) in cases {
                        let new_t = *block_map
                            .get(t)
                            .ok_or_else(|| format!("Switch case target {:?} not found", t))?;
                        new_cases.push((*v, new_t));
                    }
                    let new_default = *block_map
                        .get(default)
                        .ok_or_else(|| format!("Switch default {:?} not found", default))?;
                    IrTerminator::Switch {
                        value: *reg_map.get(value).unwrap_or(value),
                        cases: new_cases,
                        default: new_default,
                    }
                }
                other => other.clone(),
            };

            // Update the new block
            if let Some(new_block) = caller.cfg.get_block_mut(new_block_id) {
                new_block.phi_nodes = new_phis;
                new_block.instructions = new_instructions;
                new_block.terminator = new_terminator;
            }
        }

        // Merge return values into the continuation block.
        // - Single return: use a Copy instruction (preserves SRA/optimizer compatibility)
        // - Multiple returns: use a phi node (correct SSA for merging values from different paths)
        if let Some(dest) = call_site.dest {
            if return_phi_incoming.len() == 1 {
                // Single return path: add Copy in the return block (like before)
                let (return_block_id, return_val) = return_phi_incoming[0];
                if let Some(ret_block) = caller.cfg.get_block_mut(return_block_id) {
                    ret_block.instructions.push(IrInstruction::Copy {
                        dest,
                        src: return_val,
                    });
                }
            } else if return_phi_incoming.len() > 1 {
                // Multiple return paths: use phi node for correct SSA merging
                if let Some(cont_block) = caller.cfg.get_block_mut(continuation_block) {
                    cont_block.phi_nodes.push(IrPhiNode {
                        dest,
                        incoming: return_phi_incoming,
                        ty: callee.signature.return_type.clone(),
                    });
                }
                // Register the phi dest type in the caller
                caller
                    .register_types
                    .insert(dest, callee.signature.return_type.clone());
            }
        }

        // Split the original block at the call site
        let call_block_id = call_site.block;
        let inlined_entry = *block_map.get(&callee.cfg.entry_block).ok_or_else(|| {
            format!(
                "Callee entry block {:?} not found in block_map",
                callee.cfg.entry_block
            )
        })?;

        if let Some(call_block) = caller.cfg.get_block_mut(call_block_id) {
            // Move instructions after the call to the continuation block
            let after_call: Vec<IrInstruction> = call_block
                .instructions
                .drain((call_site.instruction_index + 1)..)
                .collect();

            // Remove the call instruction
            call_block.instructions.remove(call_site.instruction_index);

            // Save original terminator for continuation
            let original_terminator = call_block.terminator.clone();

            // Redirect call block to inlined entry
            call_block.terminator = IrTerminator::Branch {
                target: inlined_entry,
            };

            // Set up continuation block
            if let Some(cont_block) = caller.cfg.get_block_mut(continuation_block) {
                cont_block.instructions.extend(after_call);
                cont_block.terminator = original_terminator;
            }
        }

        // Update predecessor info
        caller.cfg.connect_blocks(call_block_id, inlined_entry);

        // Update phi nodes in successor blocks: replace call_block_id with continuation_block
        // because the original terminator now lives in the continuation block
        let successor_blocks: Vec<IrBlockId> = {
            if let Some(cont_block) = caller.cfg.get_block(continuation_block) {
                match &cont_block.terminator {
                    IrTerminator::Branch { target } => vec![*target],
                    IrTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => vec![*true_target, *false_target],
                    IrTerminator::Switch { cases, default, .. } => {
                        let mut targets: Vec<IrBlockId> = cases.iter().map(|(_, t)| *t).collect();
                        targets.push(*default);
                        targets
                    }
                    _ => vec![],
                }
            } else {
                vec![]
            }
        };

        for succ_id in successor_blocks {
            if let Some(succ_block) = caller.cfg.get_block_mut(succ_id) {
                for phi in &mut succ_block.phi_nodes {
                    for (block, _) in &mut phi.incoming {
                        if *block == call_block_id {
                            *block = continuation_block;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Remap an instruction's registers and block references.
    fn remap_instruction(
        inst: &IrInstruction,
        reg_map: &BTreeMap<IrId, IrId>,
        _block_map: &BTreeMap<IrBlockId, IrBlockId>,
    ) -> IrInstruction {
        // Clone the instruction, then remap all register uses and dests
        let mut remapped = inst.clone();

        // Remap uses (operands)
        remapped.replace_uses(reg_map);

        // Remap dest register if present
        if let Some(dest) = inst.dest() {
            if let Some(&new_dest) = reg_map.get(&dest) {
                remapped.replace_dest(new_dest);
            }
        }

        remapped
    }
}

impl OptimizationPass for InliningPass {
    fn name(&self) -> &'static str {
        "inlining"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        for _iteration in 0..self.max_iterations {
            let call_graph = CallGraph::build(module);

            // Find candidate call sites for inlining
            let mut candidates: Vec<CallSite> = Vec::new();

            for site in &call_graph.call_sites {
                if let Some(callee) = module.functions.get(&site.callee) {
                    if self.cost_model.should_inline(callee, site, &call_graph) {
                        candidates.push(site.clone());
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Sort by priority: prefer loop-nested calls and smaller functions
            candidates.sort_by(|a, b| {
                // Higher loop depth = higher priority
                b.loop_depth.cmp(&a.loop_depth)
            });

            // Find the highest register ID in use
            let mut next_reg_id = 0u32;
            for function in module.functions.values() {
                for block in function.cfg.blocks.values() {
                    for phi in &block.phi_nodes {
                        next_reg_id = next_reg_id.max(phi.dest.as_u32() + 1);
                    }
                    for inst in &block.instructions {
                        if let Some(dest) = inst.dest() {
                            next_reg_id = next_reg_id.max(dest.as_u32() + 1);
                        }
                    }
                }
            }

            // Inline multiple call sites per iteration. Group by caller,
            // then within each caller pick one site per block (back-to-front).
            // Use BTreeMap for deterministic iteration order
            let mut sites_by_caller: BTreeMap<IrFunctionId, Vec<CallSite>> = BTreeMap::new();
            for candidate in candidates {
                sites_by_caller
                    .entry(candidate.caller)
                    .or_default()
                    .push(candidate);
            }
            for sites in sites_by_caller.values_mut() {
                sites.sort_by(|a, b| b.instruction_index.cmp(&a.instruction_index));
            }

            let mut any_inlined = false;
            for (_caller_id, sites) in &sites_by_caller {
                let mut inlined_blocks: BTreeSet<IrBlockId> = BTreeSet::new();
                for candidate in sites {
                    if inlined_blocks.contains(&candidate.block) {
                        continue;
                    }
                    match Self::inline_call_site(module, candidate, &mut next_reg_id) {
                        Ok(()) => {
                            result.modified = true;
                            any_inlined = true;
                            inlined_blocks.insert(candidate.block);
                            *result
                                .stats
                                .entry("functions_inlined".to_string())
                                .or_insert(0) += 1;
                        }
                        Err(_) => {}
                    }
                }
            }
            if !any_inlined {
                break;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_model_small_function() {
        let cost_model = InliningCostModel::default();

        // A function with 3 instructions should be inlined
        assert!(3.0 <= cost_model.max_inline_size as f64);
    }

    #[test]
    fn test_cost_model_loop_bonus() {
        let cost_model = InliningCostModel::default();

        // Loop depth bonus should increase threshold
        let base_threshold = cost_model.max_inline_size as f64;
        let loop_threshold = base_threshold * cost_model.loop_depth_bonus;

        assert!(loop_threshold > base_threshold);
    }
}
