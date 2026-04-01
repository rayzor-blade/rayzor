//! Bounds Check Elimination (BCE) Pass
//!
//! Eliminates redundant array bounds checks in loops where the index is provably
//! in-bounds. Replaces `CallDirect(haxe_array_get_ptr, [arr, idx])` with inline
//! pointer arithmetic when the loop header already guarantees `idx < arr.len`.
//!
//! ## Target Pattern: for-in loops
//!
//! ```text
//! loop_cond:
//!   $idx = load $idx_ptr
//!   $cmp = cmp lt $idx, $arr_len
//!   cond_branch $cmp, loop_body, loop_exit
//!
//! loop_body:
//!   $idx2 = load $idx_ptr            // same stack slot, no store before this
//!   $ptr  = call haxe_array_get_ptr($arr, $idx2)  // <-- eliminated
//!   $val  = load $ptr
//! ```
//!
//! The header proves `$idx < $arr_len`. Since `$idx2` is loaded from the same
//! stack slot before any store, `$idx2 == $idx`, so the bounds check is redundant.
//!
//! ## Replacement
//!
//! The call is replaced with inline pointer arithmetic:
//! ```text
//!   $data_ptr  = load($arr)            // arr.ptr at offset 0
//!   $off24     = const 24
//!   $es_ptr    = add($arr, $off24)
//!   $elem_size = load($es_ptr)         // arr.elem_size at offset 24
//!   $byte_off  = mul($idx2, $elem_size)
//!   $ptr       = add($data_ptr, $byte_off)
//! ```
//!
//! After BCE, LICM hoists the invariant `$data_ptr` and `$elem_size` loads
//! out of the loop, leaving just `mul + add` per iteration.

use super::blocks::{IrBlockId, IrTerminator};
use super::functions::IrFunctionId;
use super::instructions::{BinaryOp, CompareOp, IrInstruction, OwnershipMode};
use super::loop_analysis::{DominatorTree, LoopNestInfo, NaturalLoop};
use super::optimization::{OptimizationPass, OptimizationResult};
use super::types::{IrType, IrValue};
use super::{IrFunction, IrId, IrModule};
use std::collections::{BTreeMap, BTreeSet};

pub struct BoundsCheckEliminationPass;

impl BoundsCheckEliminationPass {
    pub fn new() -> Self {
        BoundsCheckEliminationPass
    }
}

impl OptimizationPass for BoundsCheckEliminationPass {
    fn name(&self) -> &'static str {
        "BoundsCheckElimination"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut total_eliminated = 0;

        // Find the function ID for haxe_array_get_ptr
        let get_ptr_ids: BTreeSet<IrFunctionId> = module
            .extern_functions
            .iter()
            .filter(|(_, ef)| ef.name == "haxe_array_get_ptr")
            .map(|(&id, _)| id)
            .collect();

        // Also check internal functions (in case it was internalized)
        let get_ptr_ids: BTreeSet<IrFunctionId> = get_ptr_ids
            .into_iter()
            .chain(
                module
                    .functions
                    .iter()
                    .filter(|(_, f)| f.name == "haxe_array_get_ptr")
                    .map(|(&id, _)| id),
            )
            .collect();

        if get_ptr_ids.is_empty() {
            return OptimizationResult::unchanged();
        }

        // Also find array mutation functions to check array invariance
        let mutation_fn_ids: BTreeSet<IrFunctionId> = module
            .extern_functions
            .iter()
            .filter(|(_, ef)| {
                ef.name.starts_with("haxe_array_set")
                    || ef.name == "haxe_array_push"
                    || ef.name == "haxe_array_pop"
                    || ef.name == "haxe_array_insert"
                    || ef.name == "haxe_array_remove"
                    || ef.name == "haxe_array_splice"
                    || ef.name == "haxe_array_resize"
            })
            .map(|(&id, _)| id)
            .collect();

        let func_ids: Vec<_> = module.functions.keys().cloned().collect();
        for func_id in func_ids {
            if let Some(function) = module.functions.get_mut(&func_id) {
                total_eliminated +=
                    eliminate_bounds_checks(function, &get_ptr_ids, &mutation_fn_ids);
            }
        }

        if total_eliminated > 0 {
            OptimizationResult {
                modified: true,
                instructions_eliminated: total_eliminated,
                blocks_eliminated: 0,
                stats: {
                    let mut s = BTreeMap::new();
                    s.insert("bounds_checks_eliminated".to_string(), total_eliminated);
                    s
                },
            }
        } else {
            OptimizationResult::unchanged()
        }
    }
}

/// How the loop index is represented.
enum IndexSource {
    /// Pre-optimization: index loaded from a stack-allocated pointer
    StackSlot(IrId),
    /// Post-optimization: index is a phi node in the header
    Phi(IrId),
}

/// Information about the loop header's bounds-checking pattern.
struct HeaderBoundsInfo {
    /// How the index is represented
    idx_source: IndexSource,
    /// The register compared against the length in the header
    idx_cmp_reg: IrId,
    /// The register holding the array length
    len_reg: IrId,
    /// True target of the CondBranch (body block, inside loop)
    body_entry: IrBlockId,
}

/// A candidate `haxe_array_get_ptr` call site within a loop.
struct ArrayGetCallSite {
    block_id: IrBlockId,
    inst_idx: usize,
    /// The array pointer argument
    arr_reg: IrId,
    /// The index argument
    idx_reg: IrId,
    /// The destination register (result of the call)
    dest_reg: IrId,
}

/// Eliminate bounds checks in a single function. Returns count of eliminated checks.
fn eliminate_bounds_checks(
    function: &mut IrFunction,
    get_ptr_ids: &BTreeSet<IrFunctionId>,
    mutation_fn_ids: &BTreeSet<IrFunctionId>,
) -> usize {
    if function.cfg.blocks.len() < 3 {
        return 0; // Need at least preheader + header + body
    }

    let domtree = DominatorTree::compute(function);
    let loop_info = LoopNestInfo::analyze(function, &domtree);

    if loop_info.loops.is_empty() {
        return 0;
    }

    // Build a map from IrId to (block, instruction_index) for definition lookup
    let def_map = build_def_map(function);

    let mut total_eliminated = 0;

    // Process each loop (innermost first via nesting depth)
    let mut loops: Vec<&NaturalLoop> = loop_info.loops.values().collect();
    loops.sort_by(|a, b| b.nesting_depth.cmp(&a.nesting_depth));

    for natural_loop in loops {
        // Step 1: Analyze the loop header for the for-in pattern
        let header_info = match analyze_loop_header(function, natural_loop, &def_map) {
            Some(info) => info,
            None => continue,
        };

        // Step 2: Find all haxe_array_get_ptr calls in the loop body
        let candidates = find_array_get_calls(function, natural_loop, get_ptr_ids);

        if candidates.is_empty() {
            continue;
        }

        // Step 3: For each candidate, verify it's safe to eliminate
        let mut safe_sites: Vec<ArrayGetCallSite> = Vec::new();

        for call in candidates {
            // The index must match the one checked in the header
            if !is_index_proven_in_bounds(function, &call, &header_info, &def_map) {
                continue;
            }

            // For stack-slot patterns, verify length comes from the same array
            if matches!(header_info.idx_source, IndexSource::StackSlot(_)) {
                if !is_length_from_array(function, &call, &header_info, &def_map) {
                    continue;
                }
            }

            // The array must not be mutated within the loop
            if !is_array_invariant(function, call.arr_reg, natural_loop, mutation_fn_ids) {
                continue;
            }

            safe_sites.push(call);
        }

        if safe_sites.is_empty() {
            continue;
        }

        // Step 4: Replace each safe call with inline pointer arithmetic
        // Process in reverse order within each block to preserve instruction indices
        safe_sites.sort_by(|a, b| {
            a.block_id
                .as_u32()
                .cmp(&b.block_id.as_u32())
                .then(b.inst_idx.cmp(&a.inst_idx)) // reverse order within block
        });

        for site in &safe_sites {
            replace_with_inline_access(function, site);
            total_eliminated += 1;
        }
    }

    total_eliminated
}

/// Map from IrId → (block_id, instruction_index) for quick definition lookup.
type DefMap = BTreeMap<IrId, (IrBlockId, usize)>;

fn build_def_map(function: &IrFunction) -> DefMap {
    let mut map = BTreeMap::new();
    for (&block_id, block) in &function.cfg.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let Some(dest) = instruction_dest(inst) {
                map.insert(dest, (block_id, idx));
            }
        }
    }
    map
}

/// Extract the destination register of an instruction, if any.
fn instruction_dest(inst: &IrInstruction) -> Option<IrId> {
    match inst {
        IrInstruction::Const { dest, .. }
        | IrInstruction::Copy { dest, .. }
        | IrInstruction::Move { dest, .. }
        | IrInstruction::Load { dest, .. }
        | IrInstruction::LoadGlobal { dest, .. }
        | IrInstruction::BinOp { dest, .. }
        | IrInstruction::UnOp { dest, .. }
        | IrInstruction::Cmp { dest, .. }
        | IrInstruction::Cast { dest, .. }
        | IrInstruction::BitCast { dest, .. }
        | IrInstruction::Alloc { dest, .. }
        | IrInstruction::GetElementPtr { dest, .. }
        | IrInstruction::Clone { dest, .. }
        | IrInstruction::BorrowImmutable { dest, .. }
        | IrInstruction::BorrowMutable { dest, .. } => Some(*dest),
        IrInstruction::CallDirect { dest, .. } => *dest,
        IrInstruction::CallIndirect { dest, .. } => *dest,
        _ => None,
    }
}

/// Analyze the loop header to extract the bounds-checking pattern.
///
/// Handles two patterns:
///
/// **Pattern 1 (stack-slot, pre-optimization):**
///   $idx = Load { ptr: $idx_ptr, ty: I64 }
///   $cmp = Cmp { Lt, $idx, $len }
///   CondBranch { $cmp, true: body, false: exit }
///
/// **Pattern 2 (phi-based, post-optimization):**
///   $idx = Phi { [outside: init, inside: next] }
///   $cmp = Cmp { Lt, $idx, $len }
///   CondBranch { $cmp, true: body, false: exit }
fn analyze_loop_header(
    function: &IrFunction,
    loop_info: &NaturalLoop,
    _def_map: &DefMap,
) -> Option<HeaderBoundsInfo> {
    let header_block = function.cfg.blocks.get(&loop_info.header)?;

    // The terminator must be a CondBranch
    let (cmp_reg, true_target, false_target) = match &header_block.terminator {
        IrTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } => (*condition, *true_target, *false_target),
        _ => return None,
    };

    // true_target should be inside loop (body), false_target outside (exit)
    if !loop_info.blocks.contains(&true_target) || loop_info.blocks.contains(&false_target) {
        return None;
    }

    // Find the Cmp instruction that defines cmp_reg
    let (idx_reg, len_reg) = find_cmp_lt_in_block(&header_block.instructions, cmp_reg)?;

    // Try Pattern 1: idx_reg defined by Load from stack slot in this block
    if let Some(idx_stack_slot) = find_load_source_in_block(&header_block.instructions, idx_reg) {
        return Some(HeaderBoundsInfo {
            idx_source: IndexSource::StackSlot(idx_stack_slot),
            idx_cmp_reg: idx_reg,
            len_reg,
            body_entry: true_target,
        });
    }

    // Try Pattern 2: idx_reg is a phi node in the header
    for phi in &header_block.phi_nodes {
        if phi.dest == idx_reg {
            // Verify it's an induction variable: one value from outside, one from inside
            let mut has_outside = false;
            let mut has_inside = false;
            for (pred_block, _val) in &phi.incoming {
                if loop_info.blocks.contains(pred_block) {
                    has_inside = true;
                } else {
                    has_outside = true;
                }
            }
            if has_outside && has_inside {
                return Some(HeaderBoundsInfo {
                    idx_source: IndexSource::Phi(idx_reg),
                    idx_cmp_reg: idx_reg,
                    len_reg,
                    body_entry: true_target,
                });
            }
        }
    }

    None
}

/// Find `Cmp { Lt, left, right }` that defines `target_reg` in the given instructions.
/// Returns (left, right) = (index_reg, length_reg).
fn find_cmp_lt_in_block(instructions: &[IrInstruction], target_reg: IrId) -> Option<(IrId, IrId)> {
    for inst in instructions {
        if let IrInstruction::Cmp {
            dest,
            op: CompareOp::Lt,
            left,
            right,
        } = inst
        {
            if *dest == target_reg {
                return Some((*left, *right));
            }
        }
    }
    None
}

/// Find the `ptr` operand of `Load { dest: target_reg, ptr, ty: I64 }` in the given block.
fn find_load_source_in_block(instructions: &[IrInstruction], target_reg: IrId) -> Option<IrId> {
    for inst in instructions {
        if let IrInstruction::Load { dest, ptr, .. } = inst {
            if *dest == target_reg {
                return Some(*ptr);
            }
        }
    }
    None
}

/// Find all `haxe_array_get_ptr` call sites within a loop's blocks.
fn find_array_get_calls(
    function: &IrFunction,
    loop_info: &NaturalLoop,
    get_ptr_ids: &BTreeSet<IrFunctionId>,
) -> Vec<ArrayGetCallSite> {
    let mut calls = Vec::new();

    for &block_id in &loop_info.blocks {
        // Skip the header — the call is in the body, not the condition
        if block_id == loop_info.header {
            continue;
        }

        if let Some(block) = function.cfg.blocks.get(&block_id) {
            for (idx, inst) in block.instructions.iter().enumerate() {
                if let IrInstruction::CallDirect {
                    dest: Some(dest),
                    func_id,
                    args,
                    ..
                } = inst
                {
                    if get_ptr_ids.contains(func_id) && args.len() == 2 {
                        calls.push(ArrayGetCallSite {
                            block_id,
                            inst_idx: idx,
                            arr_reg: args[0],
                            idx_reg: args[1],
                            dest_reg: *dest,
                        });
                    }
                }
            }
        }
    }

    calls
}

/// Check that the call's index argument is the same value proven in-bounds
/// by the loop header comparison.
///
/// For stack-slot patterns: the call's index is loaded from the same stack slot.
/// For phi patterns: the call's index IS the phi (or a Cast of it).
fn is_index_proven_in_bounds(
    function: &IrFunction,
    call: &ArrayGetCallSite,
    header: &HeaderBoundsInfo,
    def_map: &DefMap,
) -> bool {
    match &header.idx_source {
        IndexSource::StackSlot(idx_stack_slot) => {
            let block = match function.cfg.blocks.get(&call.block_id) {
                Some(b) => b,
                None => return false,
            };

            // Find the Load instruction that defines call.idx_reg
            for inst_idx in 0..call.inst_idx {
                if let IrInstruction::Load { dest, ptr, .. } = &block.instructions[inst_idx] {
                    if *dest == call.idx_reg && *ptr == *idx_stack_slot {
                        // Verify no Store to this stack slot between this Load and the call
                        let no_intervening_store = (inst_idx + 1..call.inst_idx).all(|i| {
                            !matches!(&block.instructions[i], IrInstruction::Store { ptr: store_ptr, .. } if *store_ptr == *idx_stack_slot)
                        });
                        return no_intervening_store;
                    }
                }
            }
            false
        }
        IndexSource::Phi(phi_reg) => {
            // The call's index must be the phi itself, or a Cast/Copy of it
            if call.idx_reg == *phi_reg {
                return true;
            }

            // Check if call.idx_reg is a Cast or Copy of the phi
            if let Some(&(cast_block, cast_idx)) = def_map.get(&call.idx_reg) {
                if let Some(block) = function.cfg.blocks.get(&cast_block) {
                    match &block.instructions[cast_idx] {
                        IrInstruction::Cast { src, .. } | IrInstruction::Copy { src, .. } => {
                            return *src == *phi_reg;
                        }
                        _ => {}
                    }
                }
            }
            false
        }
    }
}

/// Verify that the header's len_reg was loaded from arr_reg + 8 (the array.len field).
///
/// Traces back through the definition chain:
///   $len_reg = Load { ptr: $len_ptr, ty: I64 }
///   $len_ptr = BinOp { Add, $arr, $offset_8 }
///   $offset_8 = Const(I64(8))
fn is_length_from_array(
    function: &IrFunction,
    call: &ArrayGetCallSite,
    header: &HeaderBoundsInfo,
    def_map: &DefMap,
) -> bool {
    // Find definition of len_reg — should be a Load
    let (len_block_id, len_idx) = match def_map.get(&header.len_reg) {
        Some(loc) => *loc,
        None => return false,
    };

    let len_block = match function.cfg.blocks.get(&len_block_id) {
        Some(b) => b,
        None => return false,
    };

    let len_ptr = match &len_block.instructions[len_idx] {
        IrInstruction::Load { ptr, .. } => *ptr,
        _ => return false,
    };

    // len_ptr should be defined as arr + 8
    let (add_block_id, add_idx) = match def_map.get(&len_ptr) {
        Some(loc) => *loc,
        None => return false,
    };

    let add_block = match function.cfg.blocks.get(&add_block_id) {
        Some(b) => b,
        None => return false,
    };

    let (add_left, add_right) = match &add_block.instructions[add_idx] {
        IrInstruction::BinOp {
            op: BinaryOp::Add,
            left,
            right,
            ..
        } => (*left, *right),
        _ => return false,
    };

    // One operand should be arr_reg, the other should be Const(8)
    let (arr_candidate, offset_candidate) = if add_left == call.arr_reg {
        (add_left, add_right)
    } else if add_right == call.arr_reg {
        (add_right, add_left)
    } else {
        return false;
    };

    // Verify offset is Const(I64(8))
    let _ = arr_candidate; // we already confirmed it matches call.arr_reg
    match def_map.get(&offset_candidate) {
        Some(&(off_block_id, off_idx)) => {
            let off_block = match function.cfg.blocks.get(&off_block_id) {
                Some(b) => b,
                None => return false,
            };
            matches!(
                &off_block.instructions[off_idx],
                IrInstruction::Const {
                    value: IrValue::I64(8),
                    ..
                }
            )
        }
        None => false,
    }
}

/// Check that the array pointer is not mutated within the loop.
///
/// Specifically, no `haxe_array_set_*`, `haxe_array_push`, etc. calls
/// target this array within the loop's blocks.
fn is_array_invariant(
    function: &IrFunction,
    arr_reg: IrId,
    loop_info: &NaturalLoop,
    mutation_fn_ids: &BTreeSet<IrFunctionId>,
) -> bool {
    for &block_id in &loop_info.blocks {
        let block = match function.cfg.blocks.get(&block_id) {
            Some(b) => b,
            None => continue,
        };

        for inst in &block.instructions {
            match inst {
                // Check for mutation calls on this array
                IrInstruction::CallDirect { func_id, args, .. }
                    if mutation_fn_ids.contains(func_id) =>
                {
                    // First arg of mutation functions is the array pointer
                    if let Some(&first_arg) = args.first() {
                        if first_arg == arr_reg {
                            return false;
                        }
                    }
                }
                // Check for direct stores to the array struct itself
                // (someone writing to arr.ptr or arr.len fields)
                IrInstruction::Store { ptr, .. } => {
                    if *ptr == arr_reg {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }

    true
}

/// Replace a `haxe_array_get_ptr` call with inline pointer arithmetic.
///
/// Transforms:
///   $dest = call haxe_array_get_ptr($arr, $idx)
///
/// Into:
///   $data_ptr  = Load($arr, I64)           // arr.ptr at offset 0 (as integer)
///   $off24     = Const(I64(24))
///   $es_ptr    = BinOp(Add, $arr, $off24)
///   $elem_size = Load($es_ptr, I64)        // arr.elem_size at offset 24
///   $byte_off  = BinOp(Mul, $idx, $elem_size)
///   $dest      = BinOp(Add, $data_ptr, $byte_off)
///
/// All arithmetic uses I64. The final result is an I64 used as a pointer;
/// both backends handle int-as-ptr in Load instructions (Cranelift: i64 is
/// the pointer type; LLVM: auto-inserts inttoptr).
fn replace_with_inline_access(function: &mut IrFunction, site: &ArrayGetCallSite) {
    let arr = site.arr_reg;
    let idx = site.idx_reg;
    let dest = site.dest_reg;

    // Allocate intermediate registers
    let data_ptr_reg = function.alloc_reg();
    let elem_size_reg = function.alloc_reg();
    let idx_i64_reg = function.alloc_reg();
    let byte_off_reg = function.alloc_reg();
    // dest_reg is reused for the final pointer

    // Register types for all new registers so backends can look them up
    function.register_types.insert(data_ptr_reg, IrType::I64);
    function.register_types.insert(elem_size_reg, IrType::I64);
    function.register_types.insert(idx_i64_reg, IrType::I64);
    function.register_types.insert(byte_off_reg, IrType::I64);
    function.register_types.insert(dest, IrType::I64);

    // All Haxe array elements are 8 bytes (i64/f64/pointer slots).
    // Hardcode elem_size = 8 instead of loading it from the array struct
    // at runtime. This eliminates 2 instructions (GEP + Load for offset 24)
    // and lets LICM hoist just the data_ptr load out of the loop.
    let replacement = vec![
        // $data_ptr = Load($arr, I64)  — arr.ptr is at offset 0, loaded as integer
        IrInstruction::Load {
            dest: data_ptr_reg,
            ptr: arr,
            ty: IrType::I64,
        },
        // $elem_size = Const(8)  — all Haxe array elements are 8-byte slots
        IrInstruction::Const {
            dest: elem_size_reg,
            value: IrValue::I64(8),
        },
        // $idx_i64 = Cast($idx, I32 → I64)  — ensure index is i64 (phi may be i32)
        IrInstruction::Cast {
            dest: idx_i64_reg,
            src: idx,
            from_ty: IrType::I32,
            to_ty: IrType::I64,
        },
        // $byte_off = Mul($idx_i64, $elem_size)  — both I64
        IrInstruction::BinOp {
            dest: byte_off_reg,
            op: BinaryOp::Mul,
            left: idx_i64_reg,
            right: elem_size_reg,
        },
        // $dest = Add($data_ptr, $byte_off)  — both I64, safe for all backends
        IrInstruction::BinOp {
            dest,
            op: BinaryOp::Add,
            left: data_ptr_reg,
            right: byte_off_reg,
        },
    ];

    if let Some(block) = function.cfg.blocks.get_mut(&site.block_id) {
        // Replace the single CallDirect with the inline sequence
        block
            .instructions
            .splice(site.inst_idx..=site.inst_idx, replacement);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::blocks::*;
    use crate::ir::functions::*;
    use crate::ir::instructions::*;
    use crate::ir::modules::*;
    use crate::ir::types::*;
    use crate::ir::*;
    use crate::tast::SymbolId;

    fn make_sig(params: Vec<IrType>, return_type: IrType) -> IrFunctionSignature {
        IrFunctionSignature {
            parameters: params
                .into_iter()
                .enumerate()
                .map(|(i, ty)| IrParameter {
                    name: format!("p{}", i),
                    ty,
                    reg: IrId::new(i as u32),
                    by_ref: false,
                })
                .collect(),
            return_type,
            calling_convention: CallingConvention::C,
            can_throw: false,
            type_params: Vec::new(),
            uses_sret: false,
        }
    }

    fn make_extern(id: IrFunctionId, name: &str, sig: IrFunctionSignature) -> IrExternFunction {
        IrExternFunction {
            id,
            name: name.to_string(),
            symbol_id: SymbolId::from_raw(9999),
            signature: sig,
            source: "runtime".to_string(),
        }
    }

    /// Build a minimal module with a for-in style loop that calls haxe_array_get_ptr.
    fn build_test_module() -> IrModule {
        let mut module = IrModule::new("test".to_string(), "test.hx".to_string());

        // Register haxe_array_get_ptr as extern function
        let get_ptr_fid = IrFunctionId(50);
        module.extern_functions.insert(
            get_ptr_fid,
            make_extern(
                get_ptr_fid,
                "haxe_array_get_ptr",
                make_sig(
                    vec![IrType::Ptr(Box::new(IrType::Void)), IrType::I64],
                    IrType::Ptr(Box::new(IrType::U8)),
                ),
            ),
        );

        // Build a function with a for-in loop pattern:
        //
        // bb0 (preheader):
        //   $0 = param (arr)
        //   $1 = const 8
        //   $2 = add $0, $1          // arr + 8
        //   $3 = load $2, I64        // arr_len = arr.len
        //   $4 = alloc I64           // idx_ptr (stack slot)
        //   $5 = const 0
        //   store $4, $5             // *idx_ptr = 0
        //   branch bb1
        //
        // bb1 (header/cond):
        //   $6 = load $4, I64        // cur_idx = *idx_ptr
        //   $7 = cmp lt $6, $3       // cur_idx < arr_len
        //   cond_branch $7, bb2, bb3
        //
        // bb2 (body):
        //   $8 = load $4, I64        // idx = *idx_ptr (same value as $6)
        //   $9 = call haxe_array_get_ptr($0, $8)  // <-- should be eliminated
        //   $10 = load $9, I64       // element value
        //   $11 = load $4, I64
        //   $12 = const 1
        //   $13 = add $11, $12       // idx + 1
        //   store $4, $13            // *idx_ptr = idx + 1
        //   branch bb1
        //
        // bb3 (exit):
        //   return

        let func_id = IrFunctionId(1);
        let mut function = IrFunction::new(
            func_id,
            SymbolId::from_raw(1),
            "test_for_in".to_string(),
            make_sig(vec![IrType::Ptr(Box::new(IrType::Void))], IrType::Void),
        );
        function.next_reg_id = 0;

        // Allocate registers $0 through $13
        let r: Vec<IrId> = (0..14).map(|_| function.alloc_reg()).collect();

        let bb0 = IrBlockId::new(0);
        let bb1 = IrBlockId::new(1);
        let bb2 = IrBlockId::new(2);
        let bb3 = IrBlockId::new(3);

        function.cfg.entry_block = bb0;

        // bb0: preheader
        function.cfg.blocks.insert(
            bb0,
            IrBasicBlock {
                id: bb0,
                label: None,
                instructions: vec![
                    IrInstruction::Const {
                        dest: r[1],
                        value: IrValue::I64(8),
                    },
                    IrInstruction::BinOp {
                        dest: r[2],
                        op: BinaryOp::Add,
                        left: r[0],
                        right: r[1],
                    },
                    IrInstruction::Load {
                        dest: r[3],
                        ptr: r[2],
                        ty: IrType::I64,
                    },
                    IrInstruction::Alloc {
                        dest: r[4],
                        ty: IrType::I64,
                        count: None,
                    },
                    IrInstruction::Const {
                        dest: r[5],
                        value: IrValue::I64(0),
                    },
                    IrInstruction::Store {
                        ptr: r[4],
                        value: r[5],
                        store_ty: None,
                    },
                ],
                terminator: IrTerminator::Branch { target: bb1 },
                phi_nodes: vec![],
                source_location: IrSourceLocation::unknown(),
                predecessors: vec![],
                metadata: BlockMetadata::default(),
            },
        );

        // bb1: header
        function.cfg.blocks.insert(
            bb1,
            IrBasicBlock {
                id: bb1,
                label: None,
                instructions: vec![
                    IrInstruction::Load {
                        dest: r[6],
                        ptr: r[4],
                        ty: IrType::I64,
                    },
                    IrInstruction::Cmp {
                        dest: r[7],
                        op: CompareOp::Lt,
                        left: r[6],
                        right: r[3],
                    },
                ],
                terminator: IrTerminator::CondBranch {
                    condition: r[7],
                    true_target: bb2,
                    false_target: bb3,
                },
                phi_nodes: vec![],
                source_location: IrSourceLocation::unknown(),
                predecessors: vec![bb0, bb2],
                metadata: BlockMetadata::default(),
            },
        );

        // bb2: body
        function.cfg.blocks.insert(
            bb2,
            IrBasicBlock {
                id: bb2,
                label: None,
                instructions: vec![
                    IrInstruction::Load {
                        dest: r[8],
                        ptr: r[4],
                        ty: IrType::I64,
                    },
                    IrInstruction::CallDirect {
                        dest: Some(r[9]),
                        func_id: get_ptr_fid,
                        args: vec![r[0], r[8]],
                        arg_ownership: vec![OwnershipMode::Copy, OwnershipMode::Copy],
                        type_args: vec![],
                        is_tail_call: false,
                    },
                    IrInstruction::Load {
                        dest: r[10],
                        ptr: r[9],
                        ty: IrType::I64,
                    },
                    IrInstruction::Load {
                        dest: r[11],
                        ptr: r[4],
                        ty: IrType::I64,
                    },
                    IrInstruction::Const {
                        dest: r[12],
                        value: IrValue::I64(1),
                    },
                    IrInstruction::BinOp {
                        dest: r[13],
                        op: BinaryOp::Add,
                        left: r[11],
                        right: r[12],
                    },
                    IrInstruction::Store {
                        ptr: r[4],
                        value: r[13],
                        store_ty: None,
                    },
                ],
                terminator: IrTerminator::Branch { target: bb1 },
                phi_nodes: vec![],
                source_location: IrSourceLocation::unknown(),
                predecessors: vec![bb1],
                metadata: BlockMetadata::default(),
            },
        );

        // bb3: exit
        function.cfg.blocks.insert(
            bb3,
            IrBasicBlock {
                id: bb3,
                label: None,
                instructions: vec![],
                terminator: IrTerminator::Return { value: None },
                phi_nodes: vec![],
                source_location: IrSourceLocation::unknown(),
                predecessors: vec![bb1],
                metadata: BlockMetadata::default(),
            },
        );

        module.functions.insert(func_id, function);
        module
    }

    #[test]
    fn test_bce_eliminates_for_in_bounds_check() {
        let mut module = build_test_module();

        let mut pass = BoundsCheckEliminationPass::new();
        let result = pass.run_on_module(&mut module);

        assert!(result.modified, "BCE should have modified the module");
        assert_eq!(
            result.instructions_eliminated, 1,
            "Should eliminate exactly 1 bounds check"
        );

        // Verify the CallDirect was replaced with inline instructions
        let func = module.functions.values().next().unwrap();
        let body_block = func.cfg.blocks.get(&IrBlockId::new(2)).unwrap();

        // The body should no longer contain a CallDirect to haxe_array_get_ptr
        let has_get_ptr_call = body_block.instructions.iter().any(|inst| {
            matches!(inst, IrInstruction::CallDirect { func_id, .. }
                if func_id.0 == 50)
        });
        assert!(
            !has_get_ptr_call,
            "Body should not contain haxe_array_get_ptr call after BCE"
        );

        // Should contain Load + BinOp(Mul) + BinOp(Add) pattern
        let has_mul = body_block.instructions.iter().any(|inst| {
            matches!(
                inst,
                IrInstruction::BinOp {
                    op: BinaryOp::Mul,
                    ..
                }
            )
        });
        assert!(
            has_mul,
            "Body should contain Mul instruction for offset calculation"
        );
    }

    #[test]
    fn test_bce_preserves_mutated_array() {
        let mut module = build_test_module();

        // Add a mutation function to the module
        let set_fid = IrFunctionId(51);
        module.extern_functions.insert(
            set_fid,
            make_extern(
                set_fid,
                "haxe_array_set_i64",
                make_sig(
                    vec![
                        IrType::Ptr(Box::new(IrType::Void)),
                        IrType::I64,
                        IrType::I64,
                    ],
                    IrType::Bool,
                ),
            ),
        );

        // Add a set call to the loop body that mutates the same array (r[0])
        let func = module.functions.values_mut().next().unwrap();
        let dummy_dest = func.alloc_reg();
        let body_block = func.cfg.blocks.get_mut(&IrBlockId::new(2)).unwrap();
        body_block.instructions.push(IrInstruction::CallDirect {
            dest: Some(dummy_dest),
            func_id: set_fid,
            args: vec![IrId::new(0), IrId::new(8), IrId::new(10)],
            arg_ownership: vec![
                OwnershipMode::Copy,
                OwnershipMode::Copy,
                OwnershipMode::Copy,
            ],
            type_args: vec![],
            is_tail_call: false,
        });

        let mut pass = BoundsCheckEliminationPass::new();
        let result = pass.run_on_module(&mut module);

        assert!(
            !result.modified,
            "BCE should NOT eliminate when array is mutated in loop"
        );
    }
}
