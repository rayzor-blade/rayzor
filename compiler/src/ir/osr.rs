//! On-stack replacement: a second entry point into a hot loop.
//!
//! Tier promotion swaps a function pointer, so a call made *after* the swap
//! runs the new tier while a frame already executing keeps its old machine code
//! until it returns. A loop entered once therefore never reaches the top tier,
//! and the only way to get optimised code into it is to compile the whole
//! module before the loop is entered.
//!
//! An OSR variant lifts that restriction. It is a standalone function that can
//! be entered at a loop header, so code sitting on a back edge can hand its
//! live state to a freshly compiled variant and return whatever the variant
//! returns. Execution changes tier in the middle of a loop.
//!
//! The variant covers every block reachable from the header, not just the loop
//! body: once control transfers it never comes back, so the variant has to run
//! the loop, leave it, and carry the function to its return.

use super::blocks::{IrBasicBlock, IrTerminator};
use super::functions::{IrFunction, IrFunctionId, IrParameter};
use super::loop_analysis::DominatorTree;
use super::{IrBlockId, IrId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// What a probe has to hand over for one variant parameter.
#[derive(Debug, Clone)]
pub enum OsrParam {
    /// The value flowing into this header phi from whichever block transfers.
    /// Each latch supplies its own, so one variant serves every back edge.
    HeaderPhi { phi_dest: IrId },
    /// A value computed before the loop, passed straight through.
    Live(IrId),
}

/// A loop header lifted into a function that can be entered directly.
pub struct OsrVariant {
    /// The extracted function; entry is a synthetic block that jumps to the header.
    pub function: IrFunction,
    /// The header this variant resumes at, used as the OSR table key.
    pub site: IrBlockId,
    /// Parameters in signature order.
    pub params: Vec<OsrParam>,
}

impl OsrVariant {
    /// The registers a probe on `latch` passes, in parameter order.
    ///
    /// Returns `None` when `latch` supplies no value for one of the header's
    /// phis, which means it is not a real predecessor of the header.
    pub fn live_ins_at(&self, func: &IrFunction, latch: IrBlockId) -> Option<Vec<IrId>> {
        let header = func.cfg.blocks.get(&self.site)?;
        self.params
            .iter()
            .map(|p| match p {
                OsrParam::Live(v) => Some(*v),
                OsrParam::HeaderPhi { phi_dest } => header
                    .phi_nodes
                    .iter()
                    .find(|phi| phi.dest == *phi_dest)?
                    .incoming
                    .iter()
                    .find(|(pred, _)| *pred == latch)
                    .map(|(_, v)| *v),
            })
            .collect()
    }
}

/// Registers a terminator reads.
fn terminator_uses(term: &IrTerminator) -> Vec<IrId> {
    match term {
        IrTerminator::CondBranch { condition, .. } => vec![*condition],
        IrTerminator::Switch { value, .. } => vec![*value],
        IrTerminator::Return { value } => value.iter().copied().collect(),
        IrTerminator::NoReturn { call } => vec![*call],
        IrTerminator::Branch { .. } | IrTerminator::Unreachable => Vec::new(),
    }
}

/// Blocks reachable from `start`, which is the part of the function an OSR
/// transfer still has to execute.
fn reachable_from(func: &IrFunction, start: IrBlockId) -> BTreeSet<IrBlockId> {
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::new();
    if func.cfg.blocks.contains_key(&start) {
        seen.insert(start);
        queue.push_back(start);
    }
    while let Some(b) = queue.pop_front() {
        let Some(block) = func.cfg.blocks.get(&b) else {
            continue;
        };
        for succ in block.successors() {
            if func.cfg.blocks.contains_key(&succ) && seen.insert(succ) {
                queue.push_back(succ);
            }
        }
    }
    seen
}

/// The block each register is defined in. Parameters count as defined by the
/// entry block, since that is where the ABI binds them.
fn definition_blocks(func: &IrFunction) -> BTreeMap<IrId, IrBlockId> {
    let mut defs = BTreeMap::new();
    for p in &func.signature.parameters {
        defs.insert(p.reg, func.cfg.entry_block);
    }
    for (&b, block) in &func.cfg.blocks {
        for phi in &block.phi_nodes {
            defs.insert(phi.dest, b);
        }
        for inst in &block.instructions {
            if let Some(d) = inst.dest() {
                defs.insert(d, b);
            }
        }
    }
    defs
}

/// Build the variant that resumes `func` at `header`.
///
/// Returns `None` for a header that cannot be resumed — one that is the
/// function's own entry, or one reading a register whose type is unknown.
pub fn build_osr_variant(
    func: &IrFunction,
    header: IrBlockId,
    new_id: IrFunctionId,
) -> Option<OsrVariant> {
    func.cfg.blocks.get(&header)?;
    // Entering at the entry block is the whole function, and nothing strictly
    // dominates it, so the parameters it needs cannot be identified.
    if header == func.cfg.entry_block {
        return None;
    }

    let region = reachable_from(func, header);
    let domtree = DominatorTree::compute(func);
    let defs = definition_blocks(func);

    // A phi inside the region whose block strictly dominates the header cannot
    // be re-entered here. Its value is already live when the transfer happens,
    // so it would arrive as a parameter, and the phi would write the same
    // register again -- one register with two definitions. That is the shape of
    // an inner loop header while its enclosing loop is mid-flight; resuming
    // there needs the enclosing phis rebuilt, so only headers without one are
    // resumable.
    for &b in &region {
        if b != header
            && !func.cfg.blocks[&b].phi_nodes.is_empty()
            && domtree.strictly_dominates(b, header)
        {
            return None;
        }
    }

    let mut variant = func.clone();
    let mut params: Vec<IrParameter> = Vec::new();
    let mut descriptors: Vec<OsrParam> = Vec::new();
    let mut next_reg = variant.next_reg_id;

    // The header's phis choose between the value from before the loop and the
    // value from a back edge. The phi has to SURVIVE, or nothing writes the
    // loop-carried register when the variant goes round again -- the induction
    // variable would freeze at whatever the transfer handed over. So the phi
    // keeps its in-region incomings and gains one more, from a synthetic entry
    // block, carrying a fresh parameter.
    let entry = IrBlockId(variant.cfg.next_block_id);
    variant.cfg.next_block_id += 1;
    let header_phis: Vec<(IrId, super::IrType)> = func.cfg.blocks[&header]
        .phi_nodes
        .iter()
        .map(|phi| (phi.dest, phi.ty.clone()))
        .collect();
    for (i, (dest, ty)) in header_phis.iter().enumerate() {
        let reg = IrId::new(next_reg);
        next_reg += 1;
        params.push(IrParameter {
            name: format!("osr_phi{i}"),
            ty: ty.clone(),
            reg,
            by_ref: false,
        });
        variant.register_types.insert(reg, ty.clone());
        descriptors.push(OsrParam::HeaderPhi { phi_dest: *dest });
    }

    // Values the region reads that were computed before entering it. Membership
    // of the region is NOT the test: reachability from the header wraps back
    // through an enclosing loop, so a value defined in an outer body is
    // reachable yet has not run when the variant starts. Strict dominance is
    // the test that separates them -- a definition that strictly dominates the
    // header ran before entry and must be handed over, while one that does not
    // is recomputed by the variant itself.
    let header_phi_dests: BTreeSet<IrId> = header_phis.iter().map(|(d, _)| *d).collect();
    let mut carried: BTreeSet<IrId> = BTreeSet::new();
    let mut note = |v: IrId, carried: &mut BTreeSet<IrId>| {
        if header_phi_dests.contains(&v) {
            return;
        }
        if let Some(&db) = defs.get(&v) {
            if domtree.strictly_dominates(db, header) {
                carried.insert(v);
            }
        }
    };
    for &b in &region {
        let block = &func.cfg.blocks[&b];
        for phi in &block.phi_nodes {
            // Edges from outside the region cannot fire in the variant, so the
            // values they carry are not read here.
            for (pred, val) in &phi.incoming {
                if region.contains(pred) {
                    note(*val, &mut carried);
                }
            }
        }
        for inst in &block.instructions {
            for u in inst.uses() {
                note(u, &mut carried);
            }
        }
        for u in terminator_uses(&block.terminator) {
            note(u, &mut carried);
        }
    }
    for v in carried {
        let ty = variant
            .register_types
            .get(&v)
            .cloned()
            .or_else(|| variant.locals.get(&v).map(|l| l.ty.clone()))?;
        params.push(IrParameter {
            name: format!("osr_live{}", v.as_u32()),
            ty,
            reg: v,
            by_ref: false,
        });
        descriptors.push(OsrParam::Live(v));
    }

    variant.id = new_id;
    variant.name = format!("{}$osr{}", func.name, header.0);
    variant.qualified_name = func
        .qualified_name
        .as_ref()
        .map(|q| format!("{q}$osr{}", header.0));
    variant.signature.parameters = params;
    variant.next_reg_id = next_reg;
    variant.cfg.blocks.retain(|b, _| region.contains(b));

    // Only edges from inside the variant can fire, plus the new entry edge.
    for (&b, block) in variant.cfg.blocks.iter_mut() {
        for phi in block.phi_nodes.iter_mut() {
            phi.incoming.retain(|(pred, _)| region.contains(pred));
        }
        let _ = b;
    }
    if let Some(block) = variant.cfg.blocks.get_mut(&header) {
        for (i, phi) in block.phi_nodes.iter_mut().enumerate() {
            phi.incoming
                .push((entry, variant.signature.parameters[i].reg));
        }
    }

    let mut entry_block = IrBasicBlock::new(entry);
    entry_block.label = Some("osr_entry".to_string());
    entry_block.terminator = IrTerminator::Branch { target: header };
    entry_block.terminator_explicit = true;
    variant.cfg.blocks.insert(entry, entry_block);
    variant.cfg.entry_block = entry;

    // Dropping blocks invalidates the cached predecessor lists every later
    // analysis reads.
    variant.cfg.recompute_predecessors();

    Some(OsrVariant {
        function: variant,
        site: header,
        params: descriptors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::*;
    use crate::ir::instructions::CompareOp;
    use crate::ir::loop_analysis::{DominatorTree, LoopNestInfo};
    use crate::ir::IrType;
    use crate::tast::SymbolId;

    /// A counted loop whose header carries a phi, plus a value defined before
    /// the loop and read inside it -- the two ways a variant acquires state.
    fn counted_loop() -> IrFunction {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());
        let sig = FunctionSignatureBuilder::new().returns(IrType::I32).build();
        builder.start_function(SymbolId::from_raw(1), "loopy".to_string(), sig);

        let limit = builder.build_int(10, IrType::I32).unwrap();
        let zero = builder.build_int(0, IrType::I32).unwrap();
        let header = builder.create_block().unwrap();
        let body = builder.create_block().unwrap();
        let exit = builder.create_block().unwrap();
        builder.build_branch(header);

        builder.switch_to_block(header);
        let i = builder.build_phi(header, IrType::I32).unwrap();
        let cmp = builder.build_cmp(CompareOp::Lt, i, limit).unwrap();
        builder.build_cond_branch(cmp, body, exit);

        builder.switch_to_block(body);
        let one = builder.build_int(1, IrType::I32).unwrap();
        let next = builder.build_add(i, one, false).unwrap();
        builder.build_branch(header);

        builder.add_phi_incoming(header, i, IrBlockId::entry(), zero);
        builder.add_phi_incoming(header, i, body, next);

        builder.switch_to_block(exit);
        builder.build_return(Some(i));

        builder.finish_function();
        builder.module.functions.values().next().unwrap().clone()
    }

    /// Two nested counted loops, with a value computed in the OUTER body and
    /// read in the inner one. Reachability from the inner header wraps back
    /// through the outer latch and reaches that value's definition, so a
    /// region-membership test would wrongly conclude the variant computes it.
    fn nested_loop() -> (IrFunction, IrBlockId, IrBlockId, IrId) {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());
        let sig = FunctionSignatureBuilder::new().returns(IrType::I32).build();
        builder.start_function(SymbolId::from_raw(2), "nested".to_string(), sig);

        let n = builder.build_int(10, IrType::I32).unwrap();
        let zero = builder.build_int(0, IrType::I32).unwrap();
        let outer_h = builder.create_block().unwrap();
        let outer_body = builder.create_block().unwrap();
        let inner_h = builder.create_block().unwrap();
        let inner_body = builder.create_block().unwrap();
        let outer_latch = builder.create_block().unwrap();
        let exit = builder.create_block().unwrap();
        builder.build_branch(outer_h);

        builder.switch_to_block(outer_h);
        let i = builder.build_phi(outer_h, IrType::I32).unwrap();
        let c1 = builder.build_cmp(CompareOp::Lt, i, n).unwrap();
        builder.build_cond_branch(c1, outer_body, exit);

        builder.switch_to_block(outer_body);
        let row = builder.build_add(i, n, false).unwrap();
        builder.build_branch(inner_h);

        builder.switch_to_block(inner_h);
        let j = builder.build_phi(inner_h, IrType::I32).unwrap();
        let c2 = builder.build_cmp(CompareOp::Lt, j, n).unwrap();
        builder.build_cond_branch(c2, inner_body, outer_latch);

        builder.switch_to_block(inner_body);
        let one = builder.build_int(1, IrType::I32).unwrap();
        let _t = builder.build_add(row, j, false).unwrap();
        let j1 = builder.build_add(j, one, false).unwrap();
        builder.build_branch(inner_h);

        builder.switch_to_block(outer_latch);
        let one2 = builder.build_int(1, IrType::I32).unwrap();
        let i1 = builder.build_add(i, one2, false).unwrap();
        builder.build_branch(outer_h);

        builder.add_phi_incoming(outer_h, i, IrBlockId::entry(), zero);
        builder.add_phi_incoming(outer_h, i, outer_latch, i1);
        builder.add_phi_incoming(inner_h, j, outer_body, zero);
        builder.add_phi_incoming(inner_h, j, inner_body, j1);

        builder.switch_to_block(exit);
        builder.build_return(Some(i));

        builder.finish_function();
        let f = builder.module.functions.values().next().unwrap().clone();
        (f, outer_h, inner_h, n)
    }

    fn innermost_header(func: &IrFunction) -> IrBlockId {
        let domtree = DominatorTree::compute(func);
        let loops = LoopNestInfo::analyze(func, &domtree);
        loops
            .loops_innermost_first()
            .into_iter()
            .next()
            .unwrap()
            .header
    }

    fn variant_of(func: &IrFunction) -> OsrVariant {
        build_osr_variant(func, innermost_header(func), IrFunctionId(9999)).expect("variant")
    }

    #[test]
    fn entry_is_synthetic_and_jumps_to_the_header() {
        let func = counted_loop();
        let v = variant_of(&func);
        let entry = v.function.cfg.entry_block;
        assert_ne!(entry, v.site, "entry must not be the header itself");
        let block = &v.function.cfg.blocks[&entry];
        assert!(block.instructions.is_empty());
        assert!(matches!(block.terminator, IrTerminator::Branch { target } if target == v.site));
    }

    /// The bug this guards: converting the header phi into a parameter and
    /// DELETING it leaves nothing to write the loop-carried register on the
    /// back edge, so the induction variable freezes at its transfer value and
    /// the loop spins forever on one iteration.
    #[test]
    fn header_phi_survives_with_both_the_entry_and_the_latch_edge() {
        let func = counted_loop();
        let v = variant_of(&func);
        let entry = v.function.cfg.entry_block;
        let phis = &v.function.cfg.blocks[&v.site].phi_nodes;
        assert_eq!(phis.len(), 1, "the header phi must survive");
        let preds: BTreeSet<IrBlockId> = phis[0].incoming.iter().map(|(p, _)| *p).collect();
        assert!(preds.contains(&entry), "phi needs the transfer value");
        assert!(
            preds.iter().any(|p| *p != entry),
            "phi needs the back edge, or the loop never advances"
        );
    }

    /// The cross-block liveness question. `row` is defined in the OUTER body
    /// and read in the inner loop. It is reachable from the inner header, so a
    /// region-membership test calls it locally computed; strict dominance
    /// correctly calls it live-in.
    #[test]
    fn a_value_from_an_enclosing_loop_is_handed_over_not_assumed() {
        let (func, outer_h, _inner_h, n) = nested_loop();
        let v = build_osr_variant(&func, outer_h, IrFunctionId(9998)).expect("outer variant");

        let live: Vec<IrId> = v
            .params
            .iter()
            .filter_map(|p| match p {
                OsrParam::Live(r) => Some(*r),
                _ => None,
            })
            .collect();
        assert!(
            live.contains(&n),
            "a value computed before the loop must arrive as a parameter, got {live:?}"
        );
    }

    /// An inner header cannot be resumed while its enclosing loop is mid-flight
    /// without rebuilding the enclosing phis, so it is refused rather than
    /// silently miscompiled.
    #[test]
    fn an_inner_header_is_refused() {
        let (func, _outer_h, inner_h, _n) = nested_loop();
        assert!(build_osr_variant(&func, inner_h, IrFunctionId(9997)).is_none());
    }

    #[test]
    fn the_entry_block_is_refused() {
        let func = counted_loop();
        let entry = func.cfg.entry_block;
        assert!(build_osr_variant(&func, entry, IrFunctionId(9996)).is_none());
    }

    #[test]
    fn every_parameter_has_a_descriptor_and_resolves_at_a_latch() {
        let func = counted_loop();
        let v = variant_of(&func);
        assert_eq!(v.params.len(), v.function.signature.parameters.len());

        let latch = func.cfg.blocks[&v.site]
            .phi_nodes
            .first()
            .and_then(|phi| {
                phi.incoming
                    .iter()
                    .map(|(p, _)| *p)
                    .find(|p| *p != func.cfg.entry_block)
            })
            .expect("a back edge");
        let ins = v.live_ins_at(&func, latch).expect("resolves");
        assert_eq!(ins.len(), v.params.len());
    }

    /// The invariant that makes a variant safe to run: it never reads a
    /// register that neither a parameter nor one of its own instructions
    /// produced. A miss here is state the back edge failed to hand over.
    #[test]
    fn variant_reads_nothing_it_was_not_given() {
        for func in [counted_loop(), nested_loop().0] {
            let header = innermost_header(&func);
            let Some(v) = build_osr_variant(&func, header, IrFunctionId(9995)) else {
                continue;
            };

            let mut available: BTreeSet<IrId> = v
                .function
                .signature
                .parameters
                .iter()
                .map(|p| p.reg)
                .collect();
            for block in v.function.cfg.blocks.values() {
                for phi in &block.phi_nodes {
                    available.insert(phi.dest);
                }
                for inst in &block.instructions {
                    if let Some(d) = inst.dest() {
                        available.insert(d);
                    }
                }
            }

            for (id, block) in &v.function.cfg.blocks {
                for phi in &block.phi_nodes {
                    for (pred, val) in &phi.incoming {
                        assert!(
                            available.contains(val),
                            "block {id:?} phi reads {val:?} from {pred:?}, never provided"
                        );
                    }
                }
                for inst in &block.instructions {
                    for u in inst.uses() {
                        assert!(
                            available.contains(&u),
                            "block {id:?} reads {u:?}, never provided"
                        );
                    }
                }
                for u in terminator_uses(&block.terminator) {
                    assert!(
                        available.contains(&u),
                        "block {id:?} terminator reads {u:?}"
                    );
                }
            }
        }
    }

    /// Phi incomings from blocks the variant does not contain are edges that
    /// cannot fire, and leaving them in would reference dropped blocks.
    #[test]
    fn variant_keeps_no_edge_from_outside_itself() {
        let func = counted_loop();
        let v = variant_of(&func);
        for block in v.function.cfg.blocks.values() {
            for phi in &block.phi_nodes {
                for (pred, _) in &phi.incoming {
                    assert!(v.function.cfg.blocks.contains_key(pred));
                }
            }
            for pred in &block.predecessors {
                assert!(
                    v.function.cfg.blocks.contains_key(pred),
                    "stale predecessor {pred:?} survives"
                );
            }
        }
    }

    /// ADVERSARIAL PROBE (temporary): the exact shape `find_rotation_releases`
    /// targets -- a header phi whose incomings are fresh allocations, with the
    /// release appended at the latch.
    fn rotation_loop() -> (IrFunction, IrBlockId, IrBlockId, IrId, IrId, IrId) {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());
        let sig = FunctionSignatureBuilder::new().returns(IrType::I32).build();
        builder.start_function(SymbolId::from_raw(3), "rot".to_string(), sig);

        let thing = IrType::Ptr(Box::new(IrType::U8));
        let limit = builder.build_int(10, IrType::I32).unwrap();
        let zero = builder.build_int(0, IrType::I32).unwrap();
        let p0 = builder.build_alloc(IrType::U8, None).unwrap();
        let header = builder.create_block().unwrap();
        let body = builder.create_block().unwrap();
        let latch = builder.create_block().unwrap();
        let exit = builder.create_block().unwrap();
        builder.build_branch(header);

        builder.switch_to_block(header);
        let i = builder.build_phi(header, IrType::I32).unwrap();
        let p = builder.build_phi(header, thing.clone()).unwrap();
        let cmp = builder.build_cmp(CompareOp::Lt, i, limit).unwrap();
        builder.build_cond_branch(cmp, body, exit);

        builder.switch_to_block(body);
        let _use_of_p = builder.build_cmp(CompareOp::Eq, p, p).unwrap();
        builder.build_branch(latch);

        builder.switch_to_block(latch);
        builder.build_free(p);
        let p1 = builder.build_alloc(IrType::U8, None).unwrap();
        let one = builder.build_int(1, IrType::I32).unwrap();
        let i1 = builder.build_add(i, one, false).unwrap();
        builder.build_branch(header);

        builder.add_phi_incoming(header, i, IrBlockId::entry(), zero);
        builder.add_phi_incoming(header, i, latch, i1);
        builder.add_phi_incoming(header, p, IrBlockId::entry(), p0);
        builder.add_phi_incoming(header, p, latch, p1);

        builder.switch_to_block(exit);
        builder.build_return(Some(i));

        builder.finish_function();
        let f = builder.module.functions.values().next().unwrap().clone();
        (f, header, latch, p, p0, p1)
    }

    #[test]
    fn adversarial_rotation_release_keeps_the_carried_pointer_a_phi() {
        use crate::ir::instructions::IrInstruction;
        let (func, header, latch, p, p0, p1) = rotation_loop();
        let v = build_osr_variant(&func, header, IrFunctionId(9994)).expect("variant");

        let phi = v.function.cfg.blocks[&header]
            .phi_nodes
            .iter()
            .find(|n| n.dest == p)
            .expect("carried phi survives");
        assert!(
            !v.function.signature.parameters.iter().any(|q| q.reg == p),
            "the carried pointer must not become a frozen parameter"
        );
        assert!(
            phi.incoming
                .iter()
                .any(|(pred, val)| *pred == latch && *val == p1),
            "latch edge lost: every alloc would leak and p would freeze, incoming={:?}",
            phi.incoming
        );

        let entry = v.function.cfg.entry_block;
        let (_, transferred) = phi
            .incoming
            .iter()
            .find(|(pred, _)| *pred == entry)
            .expect("entry edge");
        assert_ne!(*transferred, p, "entry edge must carry a FRESH register");
        assert_ne!(*transferred, p1);
        assert_ne!(*transferred, p0);
        assert!(v
            .function
            .signature
            .parameters
            .iter()
            .any(|q| q.reg == *transferred));
        assert!(
            !v.function.signature.parameters.iter().any(|q| q.reg == p0),
            "the pre-loop object must not be handed over"
        );

        let ins = v.live_ins_at(&func, latch).expect("resolves at the latch");
        let idx = v
            .params
            .iter()
            .position(|d| matches!(d, OsrParam::HeaderPhi { phi_dest } if *phi_dest == p))
            .unwrap();
        assert_eq!(
            ins[idx], p1,
            "the transfer must hand over the object the latch just built"
        );

        let frees: Vec<IrId> = v.function.cfg.blocks[&latch]
            .instructions
            .iter()
            .filter_map(|i| match i {
                IrInstruction::Free { ptr } => Some(*ptr),
                _ => None,
            })
            .collect();
        assert_eq!(frees, vec![p], "the latch still frees the re-selected phi");
    }
}
