//! Scalar Replacement of Aggregates (SRA) Pass
//!
//! Replaces heap/stack allocations that are only accessed via GEP+Load/Store
//! with individual scalar registers (one per field). This eliminates allocation
//! overhead for small structs like `Complex { re: f64, im: f64 }` that don't
//! escape the current scope.
//!
//! The pass runs after inlining so that constructor bodies and field accesses
//! are visible in the same function. Supports multi-block patterns where the
//! alloc, stores, loads, and free may be in different basic blocks.

use super::optimization::{OptimizationPass, OptimizationResult};
use super::{
    IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction, IrModule, IrPhiNode, IrType, IrValue,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub struct ScalarReplacementPass;

impl ScalarReplacementPass {
    pub fn new() -> Self {
        Self
    }
}

/// A candidate allocation that can be scalar-replaced (function-wide).
struct SraCandidate {
    alloc_dest: IrId,
    /// (block_id, instruction_index) of the Alloc
    alloc_location: (IrBlockId, usize),
    /// Maps GEP dest IrId → field index (across all blocks)
    /// BTreeMap for deterministic iteration order when creating scalar registers
    gep_map: BTreeMap<IrId, usize>,
    /// All tracked pointer IrIds (alloc dest + GEP dests + copies/casts)
    /// BTreeSet for deterministic iteration order
    tracked: BTreeSet<IrId>,
    /// (block_id, instruction_index) of Free instructions to remove
    free_locations: Vec<(IrBlockId, usize)>,
    /// Number of fields
    num_fields: usize,
    /// Types of loads per field index
    /// BTreeMap for deterministic iteration order when creating scalar registers
    field_types: BTreeMap<usize, IrType>,
    /// All GEP instruction locations to remove: (block_id, inst_idx)
    gep_locations: Vec<(IrBlockId, usize)>,
    /// All Copy/Cast locations that propagate tracked ptrs: (block_id, inst_idx)
    copy_locations: Vec<(IrBlockId, usize)>,
}

/// A phi node that merges multiple allocations and can be flattened to scalar phis.
struct PhiSraCandidate {
    /// The phi node's destination (the merged pointer)
    phi_dest: IrId,
    /// Block containing the phi
    phi_block: IrBlockId,
    /// For each incoming edge: (source_block, alloc_id, field_stores)
    /// field_stores maps field_index -> IrId of the value stored to that field
    /// BTreeMap for deterministic iteration order
    incoming_allocs: Vec<(IrBlockId, IrId, BTreeMap<usize, IrId>)>,
    /// Number of fields
    num_fields: usize,
    /// Field types (from loads)
    /// BTreeMap for deterministic iteration order
    field_types: BTreeMap<usize, IrType>,
    /// GEP map for the phi dest: maps GEP result -> field index
    /// BTreeMap for deterministic iteration order
    phi_gep_map: BTreeMap<IrId, usize>,
    /// All allocation locations to remove
    alloc_locations: Vec<(IrBlockId, usize)>,
    /// All free locations to remove
    free_locations: Vec<(IrBlockId, usize)>,
}

impl OptimizationPass for ScalarReplacementPass {
    fn name(&self) -> &'static str {
        "scalar_replacement"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        // Allow completely disabling SRA via environment variable for debugging
        if std::env::var("RAYZOR_NO_SRA").is_ok() {
            return result;
        }

        // Identify malloc and free function IDs
        let mut malloc_ids = HashSet::new();
        let mut free_ids = HashSet::new();
        for (&fid, func) in &module.functions {
            if func.name == "malloc" {
                malloc_ids.insert(fid);
            } else if func.name == "free" {
                free_ids.insert(fid);
            }
        }
        for (&fid, func) in &module.extern_functions {
            if func.name == "malloc" {
                malloc_ids.insert(fid);
            } else if func.name == "free" {
                free_ids.insert(fid);
            }
        }

        // Sort function IDs for deterministic iteration order
        let mut func_ids: Vec<_> = module.functions.keys().copied().collect();
        func_ids.sort_by_key(|id| id.0);

        for fid in func_ids {
            let function = module.functions.get_mut(&fid).unwrap();

            // Phi-aware SRA handles loop-carried allocations where pointers flow through phi nodes.
            // Enabled by default since stability fixes. Disable with RAYZOR_NO_PHI_SRA=1 if needed.
            // Run BEFORE regular SRA to avoid conflicts
            if std::env::var("RAYZOR_NO_PHI_SRA").is_err() {
                let r = run_phi_sra_on_function(function, &malloc_ids, &free_ids);
                if r.modified {
                    result.modified = true;
                    result.instructions_eliminated += r.instructions_eliminated;
                    for (k, v) in &r.stats {
                        *result.stats.entry(k.clone()).or_insert(0) += v;
                    }
                }
            }

            // Run regular SRA for non-phi allocations
            let r = run_sra_on_function(function, &malloc_ids, &free_ids);
            if r.modified {
                result.modified = true;
                result.instructions_eliminated += r.instructions_eliminated;
                for (k, v) in &r.stats {
                    *result.stats.entry(k.clone()).or_insert(0) += v;
                }
            }
        }

        result
    }
}

fn run_sra_on_function(
    function: &mut IrFunction,
    malloc_ids: &HashSet<IrFunctionId>,
    free_ids: &HashSet<IrFunctionId>,
) -> OptimizationResult {
    let mut result = OptimizationResult::unchanged();

    let constants = build_constant_map(&function.cfg);
    let candidates = find_candidates_in_function(&function.cfg, &constants, malloc_ids, free_ids);

    // Process only ONE candidate per pass to avoid stale data issues.
    // When we apply SRA to one candidate, it removes/replaces instructions,
    // which shifts indices and invalidates the pre-computed free_locations
    // and alloc_location for other candidates. By processing one at a time
    // and letting the optimizer loop re-run this pass, we ensure each
    // candidate is analyzed on the current function state.
    if let Some(candidate) = candidates.first() {
        let eliminated = apply_sra(function, candidate);
        if eliminated > 0 {
            result.modified = true;
            result.instructions_eliminated += eliminated;
            *result
                .stats
                .entry("allocs_replaced".to_string())
                .or_insert(0) += 1;
        }
    }

    result
}

/// Phi-aware SRA: handles loop-carried allocations where multiple allocations
/// flow into a phi node. Replaces pointer phis with scalar phis for each field.
fn run_phi_sra_on_function(
    function: &mut IrFunction,
    malloc_ids: &HashSet<IrFunctionId>,
    free_ids: &HashSet<IrFunctionId>,
) -> OptimizationResult {
    let mut result = OptimizationResult::unchanged();

    let constants = build_constant_map(&function.cfg);
    let candidates = find_phi_sra_candidates(&function.cfg, &constants, malloc_ids, free_ids);

    // Process only ONE candidate per pass to avoid stale data issues.
    // When we apply phi-SRA to one candidate, it modifies the function structure,
    // which can invalidate the analysis for other candidates. By processing one
    // at a time and letting the optimizer loop re-run this pass, we ensure each
    // candidate is analyzed on the current function state.
    if let Some(candidate) = candidates.into_iter().next() {
        let eliminated = apply_phi_sra(function, &candidate);
        if eliminated > 0 {
            result.modified = true;
            result.instructions_eliminated += eliminated;
            *result
                .stats
                .entry("phi_allocs_replaced".to_string())
                .or_insert(0) += 1;
        }
    }

    result
}

/// Trace a value back through Copy chains to find the original source.
/// Returns the original IrId (which may be the same as input if not a copy).
fn trace_copy_chain(id: IrId, cfg: &super::blocks::IrControlFlowGraph) -> IrId {
    let mut current = id;
    let mut visited = HashSet::new();
    let sorted = sorted_blocks(cfg);

    while visited.insert(current) {
        let mut found_copy = false;
        for &(_, block) in &sorted {
            for inst in &block.instructions {
                if let IrInstruction::Copy { dest, src } = inst {
                    if *dest == current {
                        current = *src;
                        found_copy = true;
                        break;
                    }
                }
            }
            if found_copy {
                break;
            }
        }
        if !found_copy {
            break;
        }
    }
    current
}

/// Get blocks sorted by ID for deterministic iteration order.
fn sorted_blocks(
    cfg: &super::blocks::IrControlFlowGraph,
) -> Vec<(IrBlockId, &super::blocks::IrBasicBlock)> {
    let mut blocks: Vec<_> = cfg.blocks.iter().map(|(&id, b)| (id, b)).collect();
    blocks.sort_by_key(|(id, _)| id.0);
    blocks
}

/// Find phi nodes that merge allocations and can be flattened.
fn find_phi_sra_candidates(
    cfg: &super::blocks::IrControlFlowGraph,
    constants: &HashMap<IrId, i64>,
    malloc_ids: &HashSet<IrFunctionId>,
    free_ids: &HashSet<IrFunctionId>,
) -> Vec<PhiSraCandidate> {
    let mut candidates = Vec::new();
    let sorted = sorted_blocks(cfg);

    // Build map of IrId -> (block_id, inst_idx) for all malloc calls
    let mut malloc_locations: HashMap<IrId, (IrBlockId, usize)> = HashMap::new();
    for &(block_id, block) in &sorted {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let IrInstruction::CallDirect {
                dest: Some(dest),
                func_id,
                args,
                ..
            } = inst
            {
                if malloc_ids.contains(func_id) && args.len() == 1 {
                    malloc_locations.insert(*dest, (block_id, idx));
                }
            }
            if let IrInstruction::Alloc {
                dest, count: None, ..
            } = inst
            {
                malloc_locations.insert(*dest, (block_id, idx));
            }
        }
    }

    // Find phi nodes where all incoming values are malloc results (or copies thereof)
    for &(phi_block, block) in &sorted {
        for phi in &block.phi_nodes {
            // Only consider pointer-typed phis
            if !matches!(&phi.ty, IrType::Ptr(_)) {
                continue;
            }

            // Check if all incoming values are either mallocs or the phi itself (back edge)
            let mut all_mallocs = true;
            let mut has_back_edge = false;
            let mut incoming_mallocs: Vec<(IrBlockId, IrId)> = Vec::new();

            for (src_block, value) in &phi.incoming {
                if *value == phi.dest {
                    // Back edge - phi references itself
                    has_back_edge = true;
                } else {
                    // Trace through Copy chain to find original malloc
                    let original = trace_copy_chain(*value, cfg);
                    if malloc_locations.contains_key(&original) {
                        // Store the ORIGINAL malloc ID, not the copy
                        incoming_mallocs.push((*src_block, original));
                    } else {
                        all_mallocs = false;
                        break;
                    }
                }
            }

            // Need at least one malloc and all non-back-edge values must be mallocs
            if !all_mallocs || incoming_mallocs.is_empty() {
                continue;
            }

            // Try to build a candidate
            if let Some(candidate) = try_build_phi_candidate(
                phi,
                phi_block,
                &incoming_mallocs,
                has_back_edge,
                &malloc_locations,
                cfg,
                constants,
                free_ids,
            ) {
                candidates.push(candidate);
            }
        }
    }

    candidates
}

/// Try to build a phi SRA candidate by analyzing field accesses.
fn try_build_phi_candidate(
    phi: &IrPhiNode,
    phi_block: IrBlockId,
    incoming_mallocs: &[(IrBlockId, IrId)],
    _has_back_edge: bool,
    malloc_locations: &HashMap<IrId, (IrBlockId, usize)>,
    cfg: &super::blocks::IrControlFlowGraph,
    constants: &HashMap<IrId, i64>,
    free_ids: &HashSet<IrFunctionId>,
) -> Option<PhiSraCandidate> {
    // Build value type map for tracking types from Store instructions
    let value_types = build_value_type_map(cfg);

    // Track all pointers derived from each malloc and the phi
    // Use BTreeSet/BTreeMap for deterministic iteration order
    let mut all_tracked: BTreeSet<IrId> = BTreeSet::new();
    let mut malloc_tracked: BTreeMap<IrId, BTreeSet<IrId>> = BTreeMap::new();
    let mut phi_tracked: BTreeSet<IrId> = BTreeSet::new();

    // Initialize tracking
    phi_tracked.insert(phi.dest);
    all_tracked.insert(phi.dest);

    for (_, malloc_id) in incoming_mallocs {
        let mut tracked = BTreeSet::new();
        tracked.insert(*malloc_id);
        all_tracked.insert(*malloc_id);
        malloc_tracked.insert(*malloc_id, tracked);
    }

    // Build GEP maps for each tracked set
    // Use BTreeMap for deterministic iteration order
    let mut malloc_gep_maps: BTreeMap<IrId, BTreeMap<IrId, usize>> = BTreeMap::new();
    let mut phi_gep_map: BTreeMap<IrId, usize> = BTreeMap::new();
    // Track GEP element types to detect type conflicts at the same field index
    let mut gep_element_types: BTreeMap<usize, IrType> = BTreeMap::new();

    for malloc_id in malloc_tracked.keys() {
        malloc_gep_maps.insert(*malloc_id, BTreeMap::new());
    }

    // First: Track through related phi nodes
    // When a tracked value flows into another phi, add that phi's dest to phi_tracked.
    // This handles complex control flow from while-loop short-circuit evaluation.
    let sorted = sorted_blocks(cfg);
    let mut phi_changed = true;
    while phi_changed {
        phi_changed = false;
        for &(_, block) in &sorted {
            for other_phi in &block.phi_nodes {
                if phi_tracked.contains(&other_phi.dest) {
                    continue; // Already tracking this phi
                }
                // Check if any incoming value is tracked
                let has_tracked_incoming = other_phi
                    .incoming
                    .iter()
                    .any(|(_, v)| all_tracked.contains(v));
                if has_tracked_incoming {
                    // This phi merges a tracked allocation - add it to phi_tracked
                    phi_tracked.insert(other_phi.dest);
                    all_tracked.insert(other_phi.dest);
                    phi_changed = true;
                }
            }
        }
    }

    // Iterate to fixpoint - find all GEPs derived from mallocs and phi
    let mut changed = true;
    while changed {
        changed = false;
        for &(_, block) in &sorted {
            for inst in &block.instructions {
                if let IrInstruction::GetElementPtr {
                    dest,
                    ptr,
                    indices,
                    ty,
                } = inst
                {
                    if all_tracked.contains(dest) {
                        continue;
                    }

                    // First check if this GEP is from a tracked pointer
                    let is_from_phi = phi_tracked.contains(ptr);
                    let is_from_malloc =
                        malloc_tracked.values().any(|tracked| tracked.contains(ptr));

                    // Only process GEPs from tracked pointers
                    if !is_from_phi && !is_from_malloc {
                        continue;
                    }

                    let field_idx = match resolve_gep_field_index(indices, constants) {
                        Some(idx) => idx,
                        None => return None,
                    };

                    // Track ALL field indices including index 0 (__type_id header).
                    // See comment in try_build_candidate_function_wide for rationale.

                    // Check for type conflict: same index with different element types
                    if let Some(existing_ty) = gep_element_types.get(&field_idx) {
                        if existing_ty != ty {
                            return None;
                        }
                    } else {
                        gep_element_types.insert(field_idx, ty.clone());
                    }

                    // Add to tracking
                    if is_from_phi {
                        phi_tracked.insert(*dest);
                        phi_gep_map.insert(*dest, field_idx);
                        all_tracked.insert(*dest);
                        changed = true;
                    }

                    for (malloc_id, tracked) in &mut malloc_tracked {
                        if tracked.contains(ptr) && !tracked.contains(dest) {
                            tracked.insert(*dest);
                            malloc_gep_maps
                                .get_mut(malloc_id)
                                .unwrap()
                                .insert(*dest, field_idx);
                            all_tracked.insert(*dest);
                            changed = true;
                        }
                    }
                }

                // Track copies/casts
                if let IrInstruction::Copy { dest, src } = inst {
                    if all_tracked.contains(dest) {
                        continue;
                    }
                    if phi_tracked.contains(src) {
                        phi_tracked.insert(*dest);
                        if let Some(&idx) = phi_gep_map.get(src) {
                            phi_gep_map.insert(*dest, idx);
                        }
                        all_tracked.insert(*dest);
                        changed = true;
                    }
                    for (malloc_id, tracked) in &mut malloc_tracked {
                        if tracked.contains(src) && !tracked.contains(dest) {
                            tracked.insert(*dest);
                            if let Some(&idx) = malloc_gep_maps.get(malloc_id).unwrap().get(src) {
                                malloc_gep_maps
                                    .get_mut(malloc_id)
                                    .unwrap()
                                    .insert(*dest, idx);
                            }
                            all_tracked.insert(*dest);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Check escape conditions and collect field stores/loads
    // Use BTreeMap for deterministic iteration order
    let mut field_types: BTreeMap<usize, IrType> = BTreeMap::new();
    let mut malloc_field_stores: BTreeMap<IrId, BTreeMap<usize, IrId>> = BTreeMap::new();
    let mut free_locations: Vec<(IrBlockId, usize)> = Vec::new();

    for malloc_id in malloc_tracked.keys() {
        malloc_field_stores.insert(*malloc_id, BTreeMap::new());
    }

    for &(block_id, block) in &sorted {
        // Check phi nodes for escapes (other than the one we're analyzing)
        for other_phi in &block.phi_nodes {
            if other_phi.dest == phi.dest {
                continue; // Skip the phi we're analyzing
            }
            for (_, v) in &other_phi.incoming {
                if all_tracked.contains(v) {
                    // This phi merges a tracked allocation - part of our tracking set.
                    // We allow this because while-loop short-circuit evaluation
                    // creates multiple related phi nodes for the same allocation.
                    let _ = v; // Silence warning
                }
            }
        }

        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            match inst {
                IrInstruction::Store { ptr, value } => {
                    // Check if storing to a tracked GEP
                    if all_tracked.contains(ptr) {
                        if all_tracked.contains(value) {
                            return None; // Storing tracked pointer - escapes
                        }

                        // Track field type from the stored value (for phi_tracked GEPs)
                        if phi_tracked.contains(ptr) {
                            if let Some(&field_idx) = phi_gep_map.get(ptr) {
                                if let Some(ty) = value_types.get(value) {
                                    field_types.entry(field_idx).or_insert_with(|| ty.clone());
                                }
                            }
                        }

                        // Find which malloc this store belongs to
                        for (malloc_id, tracked) in &malloc_tracked {
                            if tracked.contains(ptr) {
                                if let Some(&field_idx) =
                                    malloc_gep_maps.get(malloc_id).unwrap().get(ptr)
                                {
                                    malloc_field_stores
                                        .get_mut(malloc_id)
                                        .unwrap()
                                        .insert(field_idx, *value);
                                }
                            }
                        }
                    } else if all_tracked.contains(value) {
                        return None; // Escapes via store
                    }
                }

                IrInstruction::Load { ptr, ty, .. } => {
                    // Track field types from loads on the phi (takes precedence over Store)
                    if phi_tracked.contains(ptr) {
                        if let Some(&field_idx) = phi_gep_map.get(ptr) {
                            field_types.insert(field_idx, ty.clone());
                        }
                    }
                }

                IrInstruction::Free { ptr } => {
                    if all_tracked.contains(ptr) {
                        free_locations.push((block_id, inst_idx));
                    }
                }

                IrInstruction::CallDirect { func_id, args, .. }
                    if free_ids.contains(func_id)
                        && args.len() == 1
                        && all_tracked.contains(&args[0]) =>
                {
                    free_locations.push((block_id, inst_idx));
                }

                IrInstruction::CallDirect {
                    args,
                    dest,
                    func_id,
                    ..
                } => {
                    // Skip if this is a malloc we're tracking
                    if let Some(d) = dest {
                        if malloc_locations.contains_key(d) {
                            continue;
                        }
                    }
                    // Skip if this is a free call (already handled above)
                    if free_ids.contains(func_id)
                        && args.len() == 1
                        && all_tracked.contains(&args[0])
                    {
                        continue;
                    }
                    // Check for escapes - pointer passed to non-inlined function
                    if args.iter().any(|a| all_tracked.contains(a)) {
                        return None;
                    }
                }

                IrInstruction::Return { value: Some(v) } if all_tracked.contains(v) => {
                    return None;
                }

                _ => {}
            }
        }

        // Check terminator
        match &block.terminator {
            super::IrTerminator::Return { value: Some(v) } if all_tracked.contains(v) => {
                return None;
            }
            _ => {}
        }
    }

    // Verify all mallocs have the same number of fields stored
    if phi_gep_map.is_empty() {
        return None; // No field GEPs found from phi
    }

    // IMPORTANT: Check that at least one phi GEP is actually used by a Load instruction.
    // If all loads have been replaced with Copy (from a previous SRA run), skip this candidate.
    let mut has_load_on_phi_gep = false;
    for &(_, block) in &sorted {
        for inst in &block.instructions {
            if let IrInstruction::Load { ptr, .. } = inst {
                if phi_gep_map.contains_key(ptr) {
                    has_load_on_phi_gep = true;
                    break;
                }
            }
        }
        if has_load_on_phi_gep {
            break;
        }
    }

    if !has_load_on_phi_gep {
        return None;
    }

    let num_fields = phi_gep_map.values().max().copied().unwrap_or(0) + 1;

    // Safety check: reject candidates where any ACCESSED field lacks a known type.
    // Only check fields actually in phi_gep_map (not 0..num_fields), because
    // fields like index 0 (__type_id header) may exist in the underlying mallocs
    // but never be accessed through the phi pointer.
    for field_idx in phi_gep_map.values() {
        if !field_types.contains_key(field_idx) {
            return None;
        }
    }

    // Verify each malloc has stores for all fields that are loaded from the phi
    for (malloc_id, stores) in &malloc_field_stores {
        for field_idx in phi_gep_map.values() {
            if !stores.contains_key(field_idx) {
                return None; // Missing field store
            }
        }
    }

    // Build incoming_allocs
    let mut incoming_allocs: Vec<(IrBlockId, IrId, BTreeMap<usize, IrId>)> = Vec::new();
    for (src_block, malloc_id) in incoming_mallocs {
        let stores = malloc_field_stores.get(malloc_id).unwrap().clone();
        incoming_allocs.push((*src_block, *malloc_id, stores));
    }

    // Collect alloc locations
    let alloc_locations: Vec<(IrBlockId, usize)> = incoming_mallocs
        .iter()
        .filter_map(|(_, malloc_id)| malloc_locations.get(malloc_id).copied())
        .collect();

    Some(PhiSraCandidate {
        phi_dest: phi.dest,
        phi_block,
        incoming_allocs,
        num_fields,
        field_types,
        phi_gep_map,
        alloc_locations,
        free_locations,
    })
}

/// Apply phi-aware SRA transformation.
fn apply_phi_sra(function: &mut IrFunction, candidate: &PhiSraCandidate) -> usize {
    let mut eliminated = 0;
    let sorted = sorted_blocks(&function.cfg);

    // Recompute next_reg_id by scanning all existing IDs to avoid conflicts
    // This is necessary because other passes might have modified the function
    let mut max_id = function.next_reg_id;
    for &(_, block) in &sorted {
        for phi in &block.phi_nodes {
            max_id = max_id.max(phi.dest.as_u32() + 1);
            for (_, v) in &phi.incoming {
                max_id = max_id.max(v.as_u32() + 1);
            }
        }
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() {
                max_id = max_id.max(dest.as_u32() + 1);
            }
        }
    }
    function.next_reg_id = max_id;

    // Only create scalar phis for fields that are actually accessed through phi GEPs
    // This avoids issues with vtable/header fields that might have different types
    let mut accessed_fields: Vec<usize> = candidate.phi_gep_map.values().copied().collect();
    accessed_fields.sort();
    accessed_fields.dedup();

    // Allocate scalar phi registers for each accessed field (in sorted order for determinism)
    let mut field_phi_regs: HashMap<usize, IrId> = HashMap::new();
    for field_idx in &accessed_fields {
        let id = IrId::new(function.next_reg_id);
        function.next_reg_id += 1;
        field_phi_regs.insert(*field_idx, id);
    }

    // Create scalar phi nodes in the phi block
    let phi_block = function.cfg.blocks.get_mut(&candidate.phi_block).unwrap();

    // Get back-edge info before mutating
    let back_edge_info: Option<IrBlockId> = phi_block
        .phi_nodes
        .iter()
        .find(|p| p.dest == candidate.phi_dest)
        .and_then(|p| {
            p.incoming
                .iter()
                .find(|(_, v)| *v == candidate.phi_dest)
                .map(|(block, _)| *block)
        });

    for &field_idx in &accessed_fields {
        let field_reg = field_phi_regs[&field_idx];
        let ty = candidate
            .field_types
            .get(&field_idx)
            .cloned()
            .unwrap_or(IrType::F64);

        // Build incoming values for the scalar phi
        let mut incoming: Vec<(IrBlockId, IrId)> = Vec::new();
        for (src_block, _malloc_id, stores) in &candidate.incoming_allocs {
            if let Some(&value) = stores.get(&field_idx) {
                incoming.push((*src_block, value));
            }
        }

        // Handle back edge - use the scalar phi itself
        if let Some(back_edge_block) = back_edge_info {
            incoming.push((back_edge_block, field_reg));
        }

        let scalar_phi = IrPhiNode {
            dest: field_reg,
            incoming,
            ty,
        };
        phi_block.phi_nodes.push(scalar_phi);
    }

    // DON'T remove the original pointer phi here - GEPs still reference it.
    // DCE will clean it up once all uses are replaced.

    // Track field loads to replace
    let mut load_replacements: HashMap<IrId, IrId> = HashMap::new();
    for (&gep_dest, &field_idx) in &candidate.phi_gep_map {
        if let Some(&scalar_reg) = field_phi_regs.get(&field_idx) {
            load_replacements.insert(gep_dest, scalar_reg);
        }
    }

    // Build set of dead pointers: malloc results that feed into the phi, plus the phi itself
    // These allocations are now replaced by scalar phis, so we can remove them
    let mut dead_pointers: BTreeSet<IrId> = BTreeSet::new();
    for (_src_block, malloc_id, _stores) in &candidate.incoming_allocs {
        dead_pointers.insert(*malloc_id);
    }
    // Also add the phi_dest since it's being replaced by scalar phis
    dead_pointers.insert(candidate.phi_dest);

    // Build constant map for resolving GEP indices
    let constants = build_constant_map(&function.cfg);

    // Expand dead_pointers to include all GEPs, Copies, Casts derived from mallocs
    // Also track field indices for GEPs so we can replace their loads
    let mut gep_field_indices: HashMap<IrId, usize> = HashMap::new();
    let mut block_order: Vec<IrBlockId> = function.cfg.blocks.keys().copied().collect();
    block_order.sort_by_key(|id| id.0);

    let mut changed = true;
    while changed {
        changed = false;
        for &block_id in &block_order {
            if let Some(block) = function.cfg.blocks.get(&block_id) {
                for inst in &block.instructions {
                    match inst {
                        IrInstruction::GetElementPtr {
                            dest, ptr, indices, ..
                        } => {
                            if dead_pointers.contains(ptr) && !dead_pointers.contains(dest) {
                                dead_pointers.insert(*dest);
                                // Try to resolve field index for this GEP
                                if let Some(field_idx) =
                                    resolve_gep_field_index(indices, &constants)
                                {
                                    gep_field_indices.insert(*dest, field_idx);
                                    // Add to load_replacements if we have a scalar reg for this field
                                    if let Some(&scalar_reg) = field_phi_regs.get(&field_idx) {
                                        load_replacements.insert(*dest, scalar_reg);
                                    }
                                }
                                changed = true;
                            }
                        }
                        IrInstruction::Copy { dest, src }
                        | IrInstruction::Cast { dest, src, .. } => {
                            if dead_pointers.contains(src) && !dead_pointers.contains(dest) {
                                dead_pointers.insert(*dest);
                                // Propagate field index and load_replacement from source
                                if let Some(&field_idx) = gep_field_indices.get(src) {
                                    gep_field_indices.insert(*dest, field_idx);
                                    if let Some(&scalar_reg) = field_phi_regs.get(&field_idx) {
                                        load_replacements.insert(*dest, scalar_reg);
                                    }
                                }
                                changed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Process each block: replace loads, remove dead stores/GEPs/mallocs/frees
    for block_id in &block_order {
        let block = match function.cfg.blocks.get_mut(block_id) {
            Some(b) => b,
            None => continue,
        };

        let old_instructions = std::mem::take(&mut block.instructions);
        let mut new_instructions = Vec::with_capacity(old_instructions.len());

        for inst in old_instructions.into_iter() {
            match &inst {
                // Replace loads from phi GEPs with Copy from scalar phi
                IrInstruction::Load { dest, ptr, .. } => {
                    if let Some(&scalar_reg) = load_replacements.get(ptr) {
                        new_instructions.push(IrInstruction::Copy {
                            dest: *dest,
                            src: scalar_reg,
                        });
                        eliminated += 1;
                        continue;
                    }
                }
                // Remove stores to dead pointers (these are now dead)
                IrInstruction::Store { ptr, .. } => {
                    if dead_pointers.contains(ptr) {
                        eliminated += 1;
                        continue;
                    }
                }
                // Remove GEPs from dead pointers
                IrInstruction::GetElementPtr { ptr, .. } => {
                    if dead_pointers.contains(ptr) {
                        eliminated += 1;
                        continue;
                    }
                }
                // Remove malloc calls that produce dead pointers
                IrInstruction::CallDirect {
                    dest: Some(dest), ..
                } => {
                    if dead_pointers.contains(dest) {
                        eliminated += 1;
                        continue;
                    }
                }
                // Remove Alloc instructions that produce dead pointers
                IrInstruction::Alloc { dest, .. } => {
                    if dead_pointers.contains(dest) {
                        eliminated += 1;
                        continue;
                    }
                }
                // Remove frees of dead pointers
                IrInstruction::Free { ptr } => {
                    if dead_pointers.contains(ptr) {
                        eliminated += 1;
                        continue;
                    }
                }
                // Remove copies/casts of dead pointers
                IrInstruction::Copy { dest, src } | IrInstruction::Cast { dest, src, .. } => {
                    if dead_pointers.contains(src) || dead_pointers.contains(dest) {
                        eliminated += 1;
                        continue;
                    }
                }
                _ => {}
            }

            new_instructions.push(inst);
        }

        block.instructions = new_instructions;
    }

    // Build set of values that are being directly removed (not derived values)
    // These are the malloc results that feed into the candidate phi
    let mut directly_removed: BTreeSet<IrId> = BTreeSet::new();
    for (_src_block, malloc_id, _stores) in &candidate.incoming_allocs {
        directly_removed.insert(*malloc_id);
    }
    directly_removed.insert(candidate.phi_dest);

    // Clean up phi nodes that reference the directly removed pointers
    // Also track any phis that become empty so we can remove Frees of them
    let mut removed_phis: BTreeSet<IrId> = BTreeSet::new();
    removed_phis.insert(candidate.phi_dest); // The main phi we're replacing

    for block_id in &block_order {
        let block = match function.cfg.blocks.get_mut(block_id) {
            Some(b) => b,
            None => continue,
        };

        for phi in &mut block.phi_nodes {
            // Skip the phi we're about to remove
            if phi.dest == candidate.phi_dest {
                continue;
            }

            // Only remove incoming edges that reference directly removed values
            // (the malloc results and the original phi dest, NOT derived GEPs/copies)
            let original_len = phi.incoming.len();
            phi.incoming
                .retain(|(_, value)| !directly_removed.contains(value));

            if phi.incoming.len() < original_len {
                eliminated += original_len - phi.incoming.len();
            }
        }

        // Remove phi nodes that have become empty and track them
        let mut empty_phi_dests = Vec::new();
        for phi in &block.phi_nodes {
            if phi.incoming.is_empty() && phi.dest != candidate.phi_dest {
                empty_phi_dests.push(phi.dest);
            }
        }
        for dest in empty_phi_dests {
            removed_phis.insert(dest);
            eliminated += 1;
        }
        block.phi_nodes.retain(|phi| !phi.incoming.is_empty());
    }

    // Add removed phis to dead_pointers so Free instructions for them get removed
    for phi_id in &removed_phis {
        dead_pointers.insert(*phi_id);
    }

    // Re-expand dead_pointers to include GEPs/Copies from the removed phis
    // Also update load_replacements for new dead GEPs
    let mut changed = true;
    while changed {
        changed = false;
        for &block_id in &block_order {
            if let Some(block) = function.cfg.blocks.get(&block_id) {
                for inst in &block.instructions {
                    match inst {
                        IrInstruction::GetElementPtr {
                            dest, ptr, indices, ..
                        } => {
                            if dead_pointers.contains(ptr) && !dead_pointers.contains(dest) {
                                dead_pointers.insert(*dest);
                                // Track field index for load replacement
                                if let Some(field_idx) =
                                    resolve_gep_field_index(indices, &constants)
                                {
                                    gep_field_indices.insert(*dest, field_idx);
                                    if let Some(&scalar_reg) = field_phi_regs.get(&field_idx) {
                                        load_replacements.insert(*dest, scalar_reg);
                                    }
                                }
                                changed = true;
                            }
                        }
                        IrInstruction::Copy { dest, src }
                        | IrInstruction::Cast { dest, src, .. } => {
                            if dead_pointers.contains(src) && !dead_pointers.contains(dest) {
                                dead_pointers.insert(*dest);
                                // Propagate field index from source
                                if let Some(&field_idx) = gep_field_indices.get(src) {
                                    gep_field_indices.insert(*dest, field_idx);
                                    if let Some(&scalar_reg) = field_phi_regs.get(&field_idx) {
                                        load_replacements.insert(*dest, scalar_reg);
                                    }
                                }
                                changed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Now re-process to remove Free instructions that reference the expanded dead_pointers
    for block_id in &block_order {
        let block = match function.cfg.blocks.get_mut(block_id) {
            Some(b) => b,
            None => continue,
        };

        let old_len = block.instructions.len();
        block.instructions.retain(|inst| match inst {
            IrInstruction::Free { ptr } => !dead_pointers.contains(ptr),
            _ => true,
        });
        if block.instructions.len() < old_len {
            eliminated += old_len - block.instructions.len();
        }
    }

    // Remove the original pointer phi
    if let Some(block) = function.cfg.blocks.get_mut(&candidate.phi_block) {
        block.phi_nodes.retain(|phi| phi.dest != candidate.phi_dest);
        eliminated += 1;
    }

    eliminated
}

/// Build a map of IrId → constant value from all Const instructions.
fn build_constant_map(cfg: &super::blocks::IrControlFlowGraph) -> HashMap<IrId, i64> {
    let mut constants = HashMap::new();
    let sorted = sorted_blocks(cfg);
    for &(_, block) in &sorted {
        for inst in &block.instructions {
            if let IrInstruction::Const { dest, value } = inst {
                let int_val = match value {
                    IrValue::I32(v) => Some(*v as i64),
                    IrValue::I64(v) => Some(*v),
                    IrValue::U32(v) => Some(*v as i64),
                    IrValue::U64(v) => Some(*v as i64),
                    _ => None,
                };
                if let Some(v) = int_val {
                    constants.insert(*dest, v);
                }
            }
        }
    }
    constants
}

/// Find all SRA candidates across the entire function.
fn find_candidates_in_function(
    cfg: &super::blocks::IrControlFlowGraph,
    constants: &HashMap<IrId, i64>,
    malloc_ids: &HashSet<IrFunctionId>,
    free_ids: &HashSet<IrFunctionId>,
) -> Vec<SraCandidate> {
    let mut candidates = Vec::new();
    let sorted = sorted_blocks(cfg);

    for &(block_id, block) in &sorted {
        for (idx, inst) in block.instructions.iter().enumerate() {
            let alloc_dest = match inst {
                // Stack alloc
                IrInstruction::Alloc {
                    dest, count: None, ..
                } => *dest,
                // Heap alloc via malloc
                IrInstruction::CallDirect {
                    dest: Some(dest),
                    func_id,
                    args,
                    ..
                } if malloc_ids.contains(func_id) && args.len() == 1 => *dest,
                _ => continue,
            };

            if let Some(candidate) = try_build_candidate_function_wide(
                alloc_dest,
                (block_id, idx),
                cfg,
                constants,
                free_ids,
            ) {
                candidates.push(candidate);
            }
        }
    }

    candidates
}

/// Build a map of IrId -> IrType from instruction definitions.
fn build_value_type_map(cfg: &super::blocks::IrControlFlowGraph) -> HashMap<IrId, IrType> {
    let mut types = HashMap::new();
    let sorted = sorted_blocks(cfg);

    for &(_, block) in &sorted {
        // Types from phi nodes
        for phi in &block.phi_nodes {
            types.insert(phi.dest, phi.ty.clone());
        }

        // Types from instructions
        for inst in &block.instructions {
            match inst {
                IrInstruction::Const { dest, value } => {
                    let ty = match value {
                        IrValue::Bool(_) => IrType::Bool,
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
                        IrValue::Null | IrValue::Void | IrValue::Undef => {
                            IrType::Ptr(Box::new(IrType::Void))
                        }
                        IrValue::String(_) => IrType::Ptr(Box::new(IrType::I8)),
                        // Complex types - skip type tracking for these
                        IrValue::Array(_)
                        | IrValue::Struct(_)
                        | IrValue::Function(_)
                        | IrValue::Closure { .. } => continue,
                    };
                    types.insert(*dest, ty);
                }
                IrInstruction::Load { dest, ty, .. } => {
                    types.insert(*dest, ty.clone());
                }
                IrInstruction::Undef { dest, ty } => {
                    types.insert(*dest, ty.clone());
                }
                IrInstruction::Cast { dest, to_ty, .. } => {
                    types.insert(*dest, to_ty.clone());
                }
                IrInstruction::BinOp { dest, left, .. } => {
                    // Binary ops preserve type from left operand
                    if let Some(ty) = types.get(left) {
                        types.insert(*dest, ty.clone());
                    }
                }
                IrInstruction::Cmp { dest, .. } => {
                    // Comparisons produce bool
                    types.insert(*dest, IrType::Bool);
                }
                IrInstruction::Copy { dest, src } => {
                    if let Some(ty) = types.get(src) {
                        types.insert(*dest, ty.clone());
                    }
                }
                IrInstruction::Select { dest, true_val, .. } => {
                    if let Some(ty) = types.get(true_val) {
                        types.insert(*dest, ty.clone());
                    }
                }
                IrInstruction::Alloc { dest, ty, .. } => {
                    types.insert(*dest, IrType::Ptr(Box::new(ty.clone())));
                }
                IrInstruction::GetElementPtr { dest, ty, .. } => {
                    types.insert(*dest, IrType::Ptr(Box::new(ty.clone())));
                }
                _ => {}
            }
        }
    }

    types
}

/// Try to build an SRA candidate by scanning ALL blocks for uses.
fn try_build_candidate_function_wide(
    alloc_dest: IrId,
    alloc_location: (IrBlockId, usize),
    cfg: &super::blocks::IrControlFlowGraph,
    constants: &HashMap<IrId, i64>,
    free_ids: &HashSet<IrFunctionId>,
) -> Option<SraCandidate> {
    // Build value type map for tracking types from Store instructions
    let value_types = build_value_type_map(cfg);

    // Phase 1: Build tracked pointer set across ALL blocks
    // Use BTreeSet/BTreeMap for deterministic iteration order
    let mut tracked = BTreeSet::new();
    tracked.insert(alloc_dest);

    let mut gep_map: BTreeMap<IrId, usize> = BTreeMap::new();
    // Track GEP element types to detect type conflicts at the same field index
    let mut gep_element_types: BTreeMap<usize, IrType> = BTreeMap::new();

    // Iterate until fixpoint — find all GEPs, copies, casts derived from alloc
    let sorted = sorted_blocks(cfg);
    let mut changed = true;
    while changed {
        changed = false;
        for &(_, block) in &sorted {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::GetElementPtr {
                        dest,
                        ptr,
                        indices,
                        ty,
                    } if tracked.contains(ptr) && !tracked.contains(dest) => {
                        if let Some(field_idx) = resolve_gep_field_index(indices, constants) {
                            // Track ALL field indices including index 0 (__type_id header).
                            // The type_id store becomes a dead Copy that DCE removes.
                            // We must track index 0 because skipping it causes the
                            // safety check (num_fields vs field_types) to reject candidates
                            // where max_field_index > 0 but field 0 has no type.
                            // Check for type conflict: same index with different element types
                            if let Some(existing_ty) = gep_element_types.get(&field_idx) {
                                if existing_ty != ty {
                                    // Type conflict at this field index - reject candidate
                                    return None;
                                }
                            } else {
                                gep_element_types.insert(field_idx, ty.clone());
                            }
                            tracked.insert(*dest);
                            gep_map.insert(*dest, field_idx);
                            changed = true;
                        } else {
                            return None; // Non-constant GEP index
                        }
                    }
                    IrInstruction::Copy { dest, src }
                        if tracked.contains(src) && !tracked.contains(dest) =>
                    {
                        tracked.insert(*dest);
                        if let Some(&field_idx) = gep_map.get(src) {
                            gep_map.insert(*dest, field_idx);
                        }
                        changed = true;
                    }
                    IrInstruction::Cast {
                        dest,
                        src,
                        from_ty,
                        to_ty,
                    } if tracked.contains(src) && !tracked.contains(dest) => {
                        // If casting a tracked pointer to a non-pointer type
                        // (e.g., Ptr→I64 for generic type erasure), the allocation
                        // escapes type-safe tracking — reject this candidate.
                        if matches!(from_ty, IrType::Ptr(_)) && !matches!(to_ty, IrType::Ptr(_)) {
                            return None;
                        }
                        tracked.insert(*dest);
                        if let Some(&field_idx) = gep_map.get(src) {
                            gep_map.insert(*dest, field_idx);
                        }
                        changed = true;
                    }
                    IrInstruction::BitCast { dest, src, ty }
                        if tracked.contains(src) && !tracked.contains(dest) =>
                    {
                        // BitCast to non-pointer type means type erasure — reject
                        if !matches!(ty, IrType::Ptr(_)) {
                            if let Some(src_ty) = value_types.get(src) {
                                if matches!(src_ty, IrType::Ptr(_)) {
                                    return None;
                                }
                            }
                        }
                        tracked.insert(*dest);
                        if let Some(&field_idx) = gep_map.get(src) {
                            gep_map.insert(*dest, field_idx);
                        }
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
    }

    // Phase 2: Check escape conditions across ALL blocks
    let mut free_locations = Vec::new();
    // Use BTreeMap for deterministic iteration order when creating scalar registers
    let mut field_types: BTreeMap<usize, IrType> = BTreeMap::new();
    let mut gep_locations = Vec::new();
    let mut copy_locations = Vec::new();

    for &(block_id, block) in &sorted {
        // Check phi nodes — allow phis where ALL incoming values involving
        // tracked pointers are tracked (the phi just aliases the same alloc).
        // The phi dest becomes tracked too.
        // Reject if only SOME incoming values are tracked (merging different allocs).
        for phi in &block.phi_nodes {
            let any_tracked_incoming = phi.incoming.iter().any(|(_, v)| tracked.contains(v));
            if any_tracked_incoming || tracked.contains(&phi.dest) {
                // For now, reject phis involving tracked pointers.
                // Multi-block SRA through phis requires inserting scalar phis
                // for each field, which is a future extension.
                return None;
            }
        }
        let _ = block_id; // suppress unused warning

        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            match inst {
                IrInstruction::Store { ptr, value } => {
                    if tracked.contains(ptr) {
                        // Store to a tracked GEP field — OK only if value is NOT tracked
                        if tracked.contains(value) {
                            return None; // Tracked pointer used as stored value — escapes
                        }
                        // Track field type from the stored value
                        if let Some(&field_idx) = gep_map.get(ptr) {
                            if let Some(ty) = value_types.get(value) {
                                field_types.entry(field_idx).or_insert_with(|| ty.clone());
                            }
                        }
                    } else if tracked.contains(value) {
                        return None; // Pointer escapes via store
                    }
                }

                IrInstruction::Load { ptr, ty, .. } => {
                    if tracked.contains(ptr) {
                        if let Some(&field_idx) = gep_map.get(ptr) {
                            // Load type takes precedence over Store type
                            field_types.insert(field_idx, ty.clone());
                        }
                    }
                }

                IrInstruction::Free { ptr } => {
                    if tracked.contains(ptr) {
                        free_locations.push((block_id, inst_idx));
                    }
                }

                // CallDirect to free with a tracked pointer
                IrInstruction::CallDirect { func_id, args, .. }
                    if free_ids.contains(func_id)
                        && args.len() == 1
                        && tracked.contains(&args[0]) =>
                {
                    free_locations.push((block_id, inst_idx));
                }

                IrInstruction::GetElementPtr { ptr, dest, .. } if tracked.contains(ptr) => {
                    gep_locations.push((block_id, inst_idx));
                    // Already handled in tracking phase
                    let _ = dest;
                }

                IrInstruction::Copy { src, .. } if tracked.contains(src) => {
                    copy_locations.push((block_id, inst_idx));
                }
                IrInstruction::Cast { src, .. } if tracked.contains(src) => {
                    copy_locations.push((block_id, inst_idx));
                }
                IrInstruction::BitCast { src, .. } if tracked.contains(src) => {
                    copy_locations.push((block_id, inst_idx));
                }

                IrInstruction::Alloc { dest, .. } if *dest == alloc_dest => {}
                // The malloc call that IS this allocation
                IrInstruction::CallDirect { dest: Some(d), .. } if *d == alloc_dest => {}
                // A free call on a tracked pointer (already recorded above)
                IrInstruction::CallDirect { func_id, args, .. }
                    if free_ids.contains(func_id)
                        && args.len() == 1
                        && tracked.contains(&args[0]) => {}
                IrInstruction::Const { .. } => {}

                _ => {
                    if uses_any_tracked(inst, &tracked) {
                        return None;
                    }
                }
            }
        }

        // Check terminator
        if terminator_uses_tracked(&block.terminator, &tracked) {
            return None;
        }
    }

    if gep_map.is_empty() {
        return None;
    }

    let num_fields = gep_map.values().copied().max().unwrap_or(0) + 1;

    // Safety check: reject candidates where any accessed field lacks a known type.
    // This prevents type mismatches when creating Undef instructions.
    for field_idx in 0..num_fields {
        if !field_types.contains_key(&field_idx) {
            // Field is accessed but we couldn't determine its type
            return None;
        }
    }

    Some(SraCandidate {
        alloc_dest,
        alloc_location,
        gep_map,
        tracked,
        free_locations,
        num_fields,
        field_types,
        gep_locations,
        copy_locations,
    })
}

/// Resolve GEP indices to a single field index.
fn resolve_gep_field_index(indices: &[IrId], constants: &HashMap<IrId, i64>) -> Option<usize> {
    match indices.len() {
        1 => {
            let idx = constants.get(&indices[0])?;
            if *idx < 0 {
                return None;
            }
            Some(*idx as usize)
        }
        2 => {
            let base = constants.get(&indices[0])?;
            if *base != 0 {
                return None;
            }
            let field = constants.get(&indices[1])?;
            if *field < 0 {
                return None;
            }
            Some(*field as usize)
        }
        _ => None,
    }
}

fn uses_any_tracked(inst: &IrInstruction, tracked: &BTreeSet<IrId>) -> bool {
    match inst {
        IrInstruction::CallDirect { args, .. } => args.iter().any(|a| tracked.contains(a)),
        IrInstruction::CallIndirect { args, func_ptr, .. } => {
            tracked.contains(func_ptr) || args.iter().any(|a| tracked.contains(a))
        }
        IrInstruction::Return { value: Some(v) } => tracked.contains(v),
        IrInstruction::MemCopy { dest, src, .. } => tracked.contains(dest) || tracked.contains(src),
        IrInstruction::StoreGlobal { value, .. } => tracked.contains(value),
        IrInstruction::MakeClosure {
            captured_values, ..
        } => captured_values.iter().any(|v| tracked.contains(v)),
        IrInstruction::Throw { exception } => tracked.contains(exception),
        IrInstruction::BinOp { left, right, .. } => {
            tracked.contains(left) || tracked.contains(right)
        }
        IrInstruction::Cmp { left, right, .. } => {
            tracked.contains(left) || tracked.contains(right)
        }
        IrInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            tracked.contains(condition) || tracked.contains(true_val) || tracked.contains(false_val)
        }
        IrInstruction::Phi { incoming, .. } => incoming.iter().any(|(v, _)| tracked.contains(v)),
        _ => false,
    }
}

fn terminator_uses_tracked(terminator: &super::IrTerminator, tracked: &BTreeSet<IrId>) -> bool {
    match terminator {
        super::IrTerminator::Return { value: Some(v) } => tracked.contains(v),
        super::IrTerminator::CondBranch { condition, .. } => tracked.contains(condition),
        super::IrTerminator::Switch { value, .. } => tracked.contains(value),
        _ => false,
    }
}

/// Apply SRA rewrite for a multi-block candidate.
///
/// Strategy: process blocks in order. For each block, track field values
/// using a per-block map. When we encounter a Store to a field GEP, record
/// the value. When we encounter a Load from a field GEP, replace with a
/// Copy from the last stored value for that field.
///
/// This works for the post-inlining pattern where stores happen before loads
/// in a linear block sequence (alloc block → constructor block → use block).
fn apply_sra(function: &mut IrFunction, candidate: &SraCandidate) -> usize {
    // Recompute next_reg_id by scanning all existing IDs to avoid conflicts.
    // This is necessary because inlining and other passes may create registers
    // without updating next_reg_id, leading to ID collisions.
    let mut max_id = function.next_reg_id;
    for block in function.cfg.blocks.values() {
        for phi in &block.phi_nodes {
            max_id = max_id.max(phi.dest.as_u32() + 1);
            for (_, v) in &phi.incoming {
                max_id = max_id.max(v.as_u32() + 1);
            }
        }
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() {
                max_id = max_id.max(dest.as_u32() + 1);
            }
        }
    }
    function.next_reg_id = max_id;

    // Allocate initial Undef registers for each field
    let mut field_regs: Vec<IrId> = Vec::with_capacity(candidate.num_fields);
    for _ in 0..candidate.num_fields {
        let id = IrId::new(function.next_reg_id);
        function.next_reg_id += 1;
        field_regs.push(id);
    }

    // Global field value tracker — initialized to the Undef registers
    let mut field_current: Vec<IrId> = field_regs.clone();
    let mut eliminated = 0;

    // Collect all locations to remove, grouped by block
    let mut to_remove: HashMap<IrBlockId, HashSet<usize>> = HashMap::new();

    // Mark alloc for removal
    to_remove
        .entry(candidate.alloc_location.0)
        .or_default()
        .insert(candidate.alloc_location.1);

    // Mark frees for removal
    for &(block_id, inst_idx) in &candidate.free_locations {
        to_remove.entry(block_id).or_default().insert(inst_idx);
    }

    // Don't remove GEPs eagerly — they may be referenced by non-SRA'd code.
    // DCE will clean them up if they become dead after the rewrite.

    // Don't remove copy/cast of tracked ptrs — they may be used as values
    // elsewhere. DCE will clean them up if they become dead.

    // Build a simple block ordering: BFS from entry to process stores before loads
    let block_order = bfs_block_order(&function.cfg);

    // First pass: find all stores to determine field values per block
    // We need to process in order so loads see the right field values.
    // Build a map of (block_id, inst_idx) → replacement instruction
    let mut replacements: HashMap<(IrBlockId, usize), IrInstruction> = HashMap::new();

    for &block_id in &block_order {
        let block = match function.cfg.blocks.get(&block_id) {
            Some(b) => b,
            None => continue,
        };

        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            match inst {
                // Replace Store via GEP → Copy to field register
                IrInstruction::Store { ptr, value } if candidate.gep_map.contains_key(ptr) => {
                    let field_idx = candidate.gep_map[ptr];
                    if field_idx < candidate.num_fields {
                        let new_reg = IrId::new(function.next_reg_id);
                        function.next_reg_id += 1;
                        replacements.insert(
                            (block_id, inst_idx),
                            IrInstruction::Copy {
                                dest: new_reg,
                                src: *value,
                            },
                        );
                        field_current[field_idx] = new_reg;
                    }
                    to_remove.entry(block_id).or_default().insert(inst_idx);
                }

                // Replace Load via GEP → Copy from current field register
                IrInstruction::Load { dest, ptr, .. } if candidate.gep_map.contains_key(ptr) => {
                    let field_idx = candidate.gep_map[ptr];
                    if field_idx < candidate.num_fields {
                        replacements.insert(
                            (block_id, inst_idx),
                            IrInstruction::Copy {
                                dest: *dest,
                                src: field_current[field_idx],
                            },
                        );
                    }
                    to_remove.entry(block_id).or_default().insert(inst_idx);
                }

                _ => {}
            }
        }
    }

    // Second pass: rewrite each block
    for &block_id in &block_order {
        let block_removes = to_remove.get(&block_id);
        let block = match function.cfg.blocks.get_mut(&block_id) {
            Some(b) => b,
            None => continue,
        };

        let has_removes = block_removes.map_or(false, |s| !s.is_empty());
        if !has_removes {
            continue;
        }

        let block_removes = block_removes.unwrap();
        let old_instructions = std::mem::take(&mut block.instructions);
        let mut new_instructions = Vec::with_capacity(old_instructions.len());

        for (idx, inst) in old_instructions.into_iter().enumerate() {
            // At alloc position, insert Undef for each field
            if (block_id, idx) == candidate.alloc_location {
                for (field_idx, reg) in field_regs.iter().enumerate() {
                    let ty = candidate
                        .field_types
                        .get(&field_idx)
                        .cloned()
                        .unwrap_or(IrType::I64);
                    new_instructions.push(IrInstruction::Undef { dest: *reg, ty });
                }
                eliminated += 1;
                continue;
            }

            if block_removes.contains(&idx) {
                if let Some(replacement) = replacements.remove(&(block_id, idx)) {
                    new_instructions.push(replacement);
                }
                // else: just remove (GEP, Free, Copy of ptr, etc.)
                eliminated += 1;
                continue;
            }

            new_instructions.push(inst);
        }

        block.instructions = new_instructions;
    }

    eliminated
}

/// BFS block ordering from entry block.
fn bfs_block_order(cfg: &super::blocks::IrControlFlowGraph) -> Vec<IrBlockId> {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    queue.push_back(cfg.entry_block);
    visited.insert(cfg.entry_block);

    while let Some(block_id) = queue.pop_front() {
        order.push(block_id);

        if let Some(block) = cfg.blocks.get(&block_id) {
            for succ in block.successors() {
                if visited.insert(succ) {
                    queue.push_back(succ);
                }
            }
        }
    }

    // Include any unreachable blocks not visited by BFS (sorted for determinism)
    let mut remaining: Vec<_> = cfg.blocks.keys().copied().collect();
    remaining.sort_by_key(|id| id.0);
    for block_id in remaining {
        if visited.insert(block_id) {
            order.push(block_id);
        }
    }

    order
}
