//! On-stack replacement: a second entry point into a hot loop.
//!
//! Tier promotion swaps a function pointer, so a call made *after* the swap
//! runs the new tier while a frame already executing keeps its old machine code
//! until it returns. A loop entered once therefore never reaches the top tier,
//! and the only way to get optimised code into it is to compile the whole
//! module before the loop is entered.
//!
//! An OSR variant lifts that restriction. It is a standalone function that
//! resumes at a loop header, so code sitting on the back edge can hand over its
//! live state and return whatever the variant returns.
//!
//! The variant covers every block reachable from the header, not just the loop
//! body: once control transfers it never comes back, so the variant has to run
//! the loop, leave it, and carry the function to its return.
//!
//! ## Live state travels in a frame
//!
//! The variant takes one parameter, the address of a frame holding the live
//! values, and its prologue loads them back out. Passing them as arguments
//! instead would force each to fit a register, and the backends do not even
//! agree on how a value is held -- a multi-field struct is a pointer to
//! Cranelift and a value to LLVM. A frame sidesteps that: the back edge stores
//! bytes, the prologue loads them, and neither has to know how the other keeps
//! the value in registers. It is also the only shape that scales, since a
//! header deep in a nest legitimately carries scores of live values.

use super::blocks::{IrBasicBlock, IrTerminator};
use super::functions::{IrFunction, IrFunctionId, IrParameter};
use super::instructions::IrInstruction;
use super::loop_analysis::DominatorTree;
use super::{IrBlockId, IrId, IrType, IrValue};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

/// Live values a single variant accepts. They travel in a frame rather than
/// registers, so a large count costs stack bytes and one copy at the moment of
/// transfer, not anything per iteration.
pub const OSR_MAX_LIVE_INS: usize = 128;

/// Why a header cannot be resumed. Reported rather than swallowed, so a caller
/// can say which loops were skipped and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OsrReject {
    /// No such block in the function.
    NoSuchHeader,
    /// Resuming at the entry is the whole function, and nothing strictly
    /// dominates it, so its live values cannot be identified.
    IsEntryBlock,
    /// More live values than one frame carries.
    TooManyLiveIns(usize),
    /// A live value's type has no known size, so it has nowhere to sit.
    LiveInHasNoLayout,
    /// A live value has no recorded type.
    LiveInTypeUnknown,
    /// A value is read that no instruction is known to define. `dest()` does
    /// not cover every instruction, so treating this as "already computed"
    /// would silently read whatever the register happened to hold.
    UnclassifiableValue,
    /// The function returns through a caller-provided buffer. A helper is
    /// reached as `(frame) -> ret` and has no buffer to write through, so it
    /// would hand back its own frame.
    ReturnsThroughBuffer,
    /// A block other than the header can also be entered from outside the
    /// resumed region, so on that edge it would redefine values the frame is
    /// supposed to supply. An enclosing loop's header is the usual case.
    RegionHasExternalEntry,
}

/// Where each live value sits in the frame the back edge hands over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OsrFrame {
    /// Byte offset of each live value, parallel to `OsrLayout::live_ins`.
    pub offsets: Vec<u32>,
    pub size: u32,
    pub align: u32,
}

impl OsrFrame {
    /// Lay out `types` in order, each at its natural alignment.
    pub fn for_types(types: &[IrType]) -> Self {
        let mut offsets = Vec::with_capacity(types.len());
        let mut cursor = 0usize;
        let mut align = 1usize;
        for ty in types {
            let a = frame_align_of(ty);
            align = align.max(a);
            cursor = cursor.div_ceil(a) * a;
            offsets.push(cursor as u32);
            cursor += frame_size_of(ty);
        }
        let align = align.max(1);
        Self {
            offsets,
            size: cursor.div_ceil(align).saturating_mul(align) as u32,
            align: align as u32,
        }
    }
}

/// Bytes a value of this type occupies in a frame. Zero means the type has no
/// layout here, which is a reason to refuse the header rather than guess.
pub fn frame_size_of(ty: &IrType) -> usize {
    match ty {
        IrType::Bool | IrType::I8 | IrType::U8 => 1,
        IrType::I16 | IrType::U16 => 2,
        IrType::I32 | IrType::U32 | IrType::F32 => 4,
        IrType::I64 | IrType::U64 | IrType::F64 => 8,
        IrType::Ptr(_)
        | IrType::Ref(_)
        | IrType::Slice(_)
        | IrType::String
        | IrType::Function { .. }
        | IrType::Any => 8,
        IrType::Array(elem, n) => frame_size_of(elem).saturating_mul(*n),
        IrType::Vector { element, count } => frame_size_of(element).saturating_mul(*count),
        IrType::Opaque { size, .. } => *size,
        // A struct crosses the boundary as the pointer the backends hold it by.
        IrType::Struct { .. } | IrType::Union { .. } => 8,
        IrType::Generic { base, .. } => frame_size_of(base),
        IrType::Void | IrType::TypeVar(_) => 0,
    }
}

/// Alignment a value of this type needs in a frame.
pub fn frame_align_of(ty: &IrType) -> usize {
    match ty {
        IrType::Array(elem, _) => frame_align_of(elem),
        IrType::Vector { element, .. } => frame_align_of(element),
        IrType::Opaque { align, .. } => (*align).max(1),
        IrType::Generic { base, .. } => frame_align_of(base),
        other => frame_size_of(other).clamp(1, 8),
    }
}

/// The site key beadie stores for a resume point.
///
/// Bits 63..16 hold the header's position among the function's loop headers and
/// bits 15..0 the live count, so a probe and the helper it dispatches to agree
/// on arity. The ordinal counts headers rather than blocks, which keeps a site
/// stable when unrelated blocks are added or removed.
#[inline]
pub fn encode_osr_site(loop_ordinal: u64, live_in_count: u16) -> u64 {
    (loop_ordinal << 16) | (live_in_count as u64)
}

/// Unpack a site key into `(loop_ordinal, live_in_count)`.
#[inline]
pub fn decode_osr_site(site: u64) -> (u64, u16) {
    (site >> 16, (site & 0xFFFF) as u16)
}

/// Loop headers in discovery order, so codegen and the runtime agree on which
/// ordinal names which header.
pub fn find_loop_headers(func: &IrFunction) -> Vec<IrBlockId> {
    let domtree = DominatorTree::compute(func);
    let mut headers = Vec::new();
    for (&b, block) in &func.cfg.blocks {
        for succ in block.successors() {
            // A back edge runs to a block that dominates its source.
            if domtree.dominates(succ, b) && !headers.contains(&succ) {
                headers.push(succ);
            }
        }
    }
    headers.sort();
    headers
}

/// Position of `header` among the function's loop headers.
pub fn loop_ordinal_of(func: &IrFunction, header: IrBlockId) -> Option<u64> {
    find_loop_headers(func)
        .iter()
        .position(|h| *h == header)
        .map(|i| i as u64)
}

// ─────────────────────────────────────────────────────────────────────────
// Helper slots
// ─────────────────────────────────────────────────────────────────────────

/// One pointer per resume point, holding the compiled helper for that site or
/// null. Generated code loads it directly at the back edge.
///
/// Storing the helper rather than a flag is what keeps the loop cheap. A flag
/// would still need a runtime lookup to find the code, and a call that returns
/// into the loop makes the register allocator treat caller-saved registers as
/// clobbered across the whole body. An armed site instead branches straight to
/// the helper and returns its result, so the loop holds no call at all.
///
/// The `Box` matters: it keeps each slot at a fixed address while the map
/// rehashes, and that address is baked into generated code.
///
/// Sites are keyed by the function's NAME, not its id. `IrFunctionId` counts
/// from zero in every module, so two functions from different modules share
/// one, and a shared id here would send a loop into another loop's helper.
/// Names are what already carry function identity between modules.
type HelperSlots = RwLock<HashMap<(String, u64), Box<AtomicU64>>>;

fn helper_slots() -> &'static HelperSlots {
    static SLOTS: OnceLock<HelperSlots> = OnceLock::new();
    SLOTS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Address of the slot for `(func, site_key)`, allocating it on first call.
///
/// Stable for the life of the process, which is what lets codegen embed it as
/// a constant instead of looking the site up every iteration.
pub fn helper_slot_addr(func: &str, site_key: u64) -> *const u8 {
    let mut slots = helper_slots().write().unwrap();
    let slot = slots
        .entry((func.to_string(), site_key))
        .or_insert_with(|| Box::new(AtomicU64::new(0)));
    (&**slot) as *const AtomicU64 as *const u8
}

/// Publish `helper` for `(func, site_key)`, after which back edges at that site
/// start transferring into it.
pub fn publish_helper(func: &str, site_key: u64, helper: *mut ()) {
    if osr_trace_enabled() {
        eprintln!("[osr] publish {func} site=0x{site_key:x} -> {helper:?}");
    }
    let mut slots = helper_slots().write().unwrap();
    slots
        .entry((func.to_string(), site_key))
        .or_insert_with(|| Box::new(AtomicU64::new(0)))
        .store(helper as u64, Ordering::Release);
}

/// The helper currently published for `(func, site_key)`, or null. Reads what
/// generated code reads.
pub fn helper_for(func: &str, site_key: u64) -> *mut () {
    helper_slots()
        .read()
        .unwrap()
        .get(&(func.to_string(), site_key))
        .map(|s| s.load(Ordering::Acquire) as *mut ())
        .unwrap_or(std::ptr::null_mut())
}

/// One counter per resume point, bumped by the dispatch path each time a loop
/// actually transfers.
///
/// Publishing a helper proves only that one was compiled. A transfer is the
/// thing worth knowing about, and without a count an A/B where nothing ever
/// transferred looks exactly like one where the transfer was free.
fn transfer_counts() -> &'static HelperSlots {
    static COUNTS: OnceLock<HelperSlots> = OnceLock::new();
    COUNTS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Address of the transfer counter for `(func, site_key)`, allocated on first
/// call and stable thereafter so generated code can bump it directly.
pub fn transfer_count_addr(func: &str, site_key: u64) -> *const u8 {
    let mut counts = transfer_counts().write().unwrap();
    let slot = counts
        .entry((func.to_string(), site_key))
        .or_insert_with(|| Box::new(AtomicU64::new(0)));
    (&**slot) as *const AtomicU64 as *const u8
}

/// How many times `(func, site_key)` has transferred.
pub fn transfer_count(func: &str, site_key: u64) -> u64 {
    transfer_counts()
        .read()
        .unwrap()
        .get(&(func.to_string(), site_key))
        .map(|s| s.load(Ordering::Acquire))
        .unwrap_or(0)
}

/// Every resume point that has transferred at least once.
pub fn transfers_taken() -> Vec<(String, u64, u64)> {
    transfer_counts()
        .read()
        .unwrap()
        .iter()
        .filter_map(|((name, site), n)| {
            let n = n.load(Ordering::Acquire);
            (n > 0).then(|| (name.clone(), *site, n))
        })
        .collect()
}

/// Whether resume points are built at all.
///
/// On, unless `RAYZOR_NO_OSR` says otherwise. A loop entered once is the case
/// tier promotion cannot reach on its own, so leaving this off means the top
/// tier never arrives for exactly the code that needed it most.
pub fn osr_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("RAYZOR_NO_OSR").is_none())
}

/// Whether tracing of OSR decisions is on.
///
/// Resolved once. `getenv` takes a process-wide lock on macOS, and a probe that
/// consulted the environment per iteration would spend most of a hot loop
/// inside it.
pub fn osr_trace_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("RAYZOR_OSR_TRACE").is_some())
}

/// What a probe hands over for one frame slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OsrParam {
    /// The value flowing into this header phi from whichever block transfers.
    /// Each latch supplies its own, so one variant serves every back edge.
    HeaderPhi { phi_dest: IrId },
    /// A value computed before the loop, stored as it stands.
    Live(IrId),
}

/// The live state a resume point needs, and where it sits in the frame.
#[derive(Debug, Clone)]
pub struct OsrLayout {
    pub header: IrBlockId,
    pub loop_ordinal: u64,
    /// What the back edge stores, in frame-slot order.
    pub params: Vec<OsrParam>,
    pub live_in_types: Vec<IrType>,
    /// Leading slots that are header phis; the rest are plain live values.
    pub phi_count: usize,
    pub frame: OsrFrame,
}

impl OsrLayout {
    pub fn site_key(&self) -> u64 {
        encode_osr_site(self.loop_ordinal, self.params.len() as u16)
    }

    /// The registers a probe on `latch` stores, in frame-slot order.
    ///
    /// Returns `None` when `latch` supplies no value for one of the header's
    /// phis, which means it is not a predecessor of the header.
    pub fn live_ins_at(&self, func: &IrFunction, latch: IrBlockId) -> Option<Vec<IrId>> {
        let header = func.cfg.blocks.get(&self.header)?;
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

/// A loop header lifted into a function that can be entered directly.
pub struct OsrVariant {
    /// The extracted function. It takes the frame address and resumes at the
    /// header; its entry block loads the live values back out.
    pub function: IrFunction,
    /// The header this variant resumes at.
    pub site: IrBlockId,
    pub layout: OsrLayout,
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
pub fn blocks_reachable_from(func: &IrFunction, start: IrBlockId) -> BTreeSet<IrBlockId> {
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

/// Blocks `header` dominates -- every path to them from the entry runs through
/// it, so a variant resuming there is guaranteed to have executed them.
pub fn blocks_dominated_by(func: &IrFunction, header: IrBlockId) -> BTreeSet<IrBlockId> {
    let domtree = DominatorTree::compute(func);
    func.cfg
        .blocks
        .keys()
        .copied()
        .filter(|b| domtree.dominates(header, *b))
        .collect()
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

/// Work out what resuming at `header` needs, or why it cannot be done.
pub fn osr_layout(func: &IrFunction, header: IrBlockId) -> Result<OsrLayout, OsrReject> {
    let header_block = func
        .cfg
        .blocks
        .get(&header)
        .ok_or(OsrReject::NoSuchHeader)?;
    if header == func.cfg.entry_block {
        return Err(OsrReject::IsEntryBlock);
    }
    if func.signature.uses_sret {
        return Err(OsrReject::ReturnsThroughBuffer);
    }

    let region = blocks_reachable_from(func, header);
    // Only blocks the header dominates are guaranteed to have run by the time
    // the resumed code reads what they define. A block that is merely reachable
    // may also be reached without the header -- an enclosing loop's header is
    // the usual case -- so anything it defines has to arrive in the frame.
    let dominated = blocks_dominated_by(func, header);
    let defs = definition_blocks(func);

    // The header's phis are the loop-carried values. They lead the frame, and
    // the walk below skips the header's own incomings: those are precisely what
    // resuming replaces.
    let mut params: Vec<OsrParam> = Vec::new();
    let mut types: Vec<IrType> = Vec::new();
    let mut seen: BTreeSet<IrId> = BTreeSet::new();
    for phi in &header_block.phi_nodes {
        params.push(OsrParam::HeaderPhi { phi_dest: phi.dest });
        types.push(phi.ty.clone());
        seen.insert(phi.dest);
    }
    let phi_count = params.len();

    let mut consider = |v: IrId,
                        params: &mut Vec<OsrParam>,
                        types: &mut Vec<IrType>,
                        seen: &mut BTreeSet<IrId>|
     -> Result<(), OsrReject> {
        if !seen.insert(v) {
            return Ok(());
        }
        let Some(&db) = defs.get(&v) else {
            // Nothing claims to define it. `dest()` does not cover every
            // instruction, so this may be a real definition we cannot see;
            // assuming it is already computed would read a stale register.
            return Err(OsrReject::UnclassifiableValue);
        };
        if dominated.contains(&db) {
            return Ok(());
        }
        let ty = func
            .register_types
            .get(&v)
            .cloned()
            .or_else(|| func.locals.get(&v).map(|l| l.ty.clone()))
            .ok_or(OsrReject::LiveInTypeUnknown)?;
        if frame_size_of(&ty) == 0 {
            return Err(OsrReject::LiveInHasNoLayout);
        }
        params.push(OsrParam::Live(v));
        types.push(ty);
        Ok(())
    };

    for &b in &region {
        let block = &func.cfg.blocks[&b];
        if b != header {
            for phi in &block.phi_nodes {
                for (pred, val) in &phi.incoming {
                    if region.contains(pred) {
                        consider(*val, &mut params, &mut types, &mut seen)?;
                    }
                }
            }
        }
        for inst in &block.instructions {
            for u in inst.uses() {
                consider(u, &mut params, &mut types, &mut seen)?;
            }
        }
        for u in terminator_uses(&block.terminator) {
            consider(u, &mut params, &mut types, &mut seen)?;
        }
    }

    if params.len() > OSR_MAX_LIVE_INS {
        return Err(OsrReject::TooManyLiveIns(params.len()));
    }

    // A block other than the header that is also entered from outside the
    // region would, on that edge, redefine values the frame supplies, and its
    // phi would take an incoming the variant has no block for.
    for &b in &region {
        if b == header {
            continue;
        }
        let block = &func.cfg.blocks[&b];
        if block
            .predecessors
            .iter()
            .any(|p| !region.contains(p) && *p != header)
        {
            return Err(OsrReject::RegionHasExternalEntry);
        }
    }

    let frame = OsrFrame::for_types(&types);
    Ok(OsrLayout {
        header,
        loop_ordinal: loop_ordinal_of(func, header).unwrap_or(u64::MAX),
        params,
        live_in_types: types,
        phi_count,
        frame,
    })
}

/// Build the variant that resumes `func` at `header`.
pub fn build_osr_variant(
    func: &IrFunction,
    header: IrBlockId,
    new_id: IrFunctionId,
) -> Result<OsrVariant, OsrReject> {
    let layout = osr_layout(func, header)?;
    let region = blocks_reachable_from(func, header);

    let mut variant = func.clone();
    let mut next_reg = variant.next_reg_id;
    let mut fresh = || {
        let r = IrId::new(next_reg);
        next_reg += 1;
        r
    };

    // One parameter: the address of the frame the back edge filled in.
    let frame_ptr = fresh();
    let frame_ty = IrType::Ptr(Box::new(IrType::I8));
    variant.register_types.insert(frame_ptr, frame_ty.clone());
    variant.signature.parameters = vec![IrParameter {
        name: "osr_frame".to_string(),
        ty: frame_ty.clone(),
        reg: frame_ptr,
        by_ref: false,
    }];
    // A probe calls the helper as `(frame) -> ret`. Left on the Haxe
    // convention the backends prepend a hidden environment pointer, which
    // would put the frame in the wrong register. Nothing is lost by dropping
    // it: an environment a resumed block reads arrives in the frame like any
    // other value, because it is a parameter and parameters are defined by the
    // entry block, which no header dominates.
    variant.signature.calling_convention = crate::ir::CallingConvention::C;

    // The prologue reads each live value back out of the frame. A phi's value
    // becomes that phi's incoming on the entry edge; everything else is bound
    // to the register it had, which the region already refers to.
    let entry = IrBlockId(variant.cfg.next_block_id);
    variant.cfg.next_block_id += 1;
    let mut prologue: Vec<IrInstruction> = Vec::new();
    let mut phi_values: Vec<IrId> = Vec::new();
    for (i, param) in layout.params.iter().enumerate() {
        let ty = layout.live_in_types[i].clone();
        let offset = fresh();
        let addr = fresh();
        let dest = match param {
            OsrParam::HeaderPhi { .. } => {
                let d = fresh();
                phi_values.push(d);
                d
            }
            OsrParam::Live(v) => *v,
        };
        variant.register_types.insert(offset, IrType::I64);
        variant.register_types.insert(addr, frame_ty.clone());
        variant.register_types.insert(dest, ty.clone());
        prologue.push(IrInstruction::Const {
            dest: offset,
            value: IrValue::I64(layout.frame.offsets[i] as i64),
        });
        prologue.push(IrInstruction::PtrAdd {
            dest: addr,
            ptr: frame_ptr,
            offset,
            ty: frame_ty.clone(),
        });
        prologue.push(IrInstruction::Load {
            dest,
            ptr: addr,
            ty,
        });
    }

    variant.id = new_id;
    variant.name = format!("{}$osr{}", func.name, header.0);
    variant.qualified_name = func
        .qualified_name
        .as_ref()
        .map(|q| format!("{q}$osr{}", header.0));
    variant.next_reg_id = next_reg;
    variant.cfg.blocks.retain(|b, _| region.contains(b));

    // Only edges from inside the variant can fire, plus the new entry edge.
    for block in variant.cfg.blocks.values_mut() {
        for phi in block.phi_nodes.iter_mut() {
            phi.incoming.retain(|(pred, _)| region.contains(pred));
        }
    }
    // The header's phis survive. Converting them to loads and deleting them
    // would leave nothing writing the loop-carried register on the back edge,
    // so the value would freeze at whatever the transfer handed over.
    if let Some(block) = variant.cfg.blocks.get_mut(&header) {
        for (i, phi) in block.phi_nodes.iter_mut().enumerate() {
            phi.incoming.push((entry, phi_values[i]));
        }
    }

    let mut entry_block = IrBasicBlock::new(entry);
    entry_block.label = Some("osr_entry".to_string());
    entry_block.instructions = prologue;
    entry_block.terminator = IrTerminator::Branch { target: header };
    entry_block.terminator_explicit = true;
    variant.cfg.blocks.insert(entry, entry_block);
    variant.cfg.entry_block = entry;

    // Dropping blocks invalidates the cached predecessor lists every later
    // analysis reads.
    variant.cfg.recompute_predecessors();

    Ok(OsrVariant {
        function: variant,
        site: header,
        layout,
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
    fn the_variant_takes_one_frame_pointer() {
        let func = counted_loop();
        let v = variant_of(&func);
        assert_eq!(
            v.function.signature.parameters.len(),
            1,
            "live state travels in the frame, not the signature"
        );
        assert!(matches!(
            v.function.signature.parameters[0].ty,
            IrType::Ptr(_)
        ));
    }

    #[test]
    fn entry_loads_every_slot_then_jumps_to_the_header() {
        let func = counted_loop();
        let v = variant_of(&func);
        let entry = v.function.cfg.entry_block;
        assert_ne!(entry, v.site);
        let block = &v.function.cfg.blocks[&entry];
        let loads = block
            .instructions
            .iter()
            .filter(|i| matches!(i, IrInstruction::Load { .. }))
            .count();
        assert_eq!(loads, v.layout.params.len(), "one load per frame slot");
        assert!(matches!(block.terminator, IrTerminator::Branch { target } if target == v.site));
    }

    /// The bug this guards: turning the header phi into a load and DELETING it
    /// leaves nothing writing the loop-carried register on the back edge, so
    /// the induction variable freezes at its transfer value.
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

    /// The cross-block liveness rule. `n` is computed before the loop, so it
    /// has to arrive in the frame. Classifying by region membership instead of
    /// dominance would call it locally computed and read it uninitialised.
    #[test]
    fn a_value_computed_before_the_loop_is_handed_over_not_assumed() {
        let (func, outer_h, _inner_h, n) = nested_loop();
        let v = build_osr_variant(&func, outer_h, IrFunctionId(9998)).expect("outer variant");
        assert!(
            v.layout.params.contains(&OsrParam::Live(n)),
            "expected {n:?} in the frame, got {:?}",
            v.layout.params
        );
    }

    /// An inner header's enclosing loop is mid-flight, so its header can be
    /// entered from outside the resumed region. Refused, with a reason.
    #[test]
    fn an_inner_header_is_refused_with_a_reason() {
        let (func, _outer_h, inner_h, _n) = nested_loop();
        assert_eq!(
            build_osr_variant(&func, inner_h, IrFunctionId(9997)).err(),
            Some(OsrReject::RegionHasExternalEntry)
        );
    }

    /// A probe calls a helper as `(frame) -> ret`, so the helper must not also
    /// be expected to write through a caller-provided buffer.
    #[test]
    fn a_function_returning_through_a_buffer_is_refused() {
        let mut func = counted_loop();
        func.signature.uses_sret = true;
        let header = innermost_header(&func);
        assert_eq!(
            build_osr_variant(&func, header, IrFunctionId(9993)).err(),
            Some(OsrReject::ReturnsThroughBuffer)
        );
    }

    /// Left on the Haxe convention the backends prepend a hidden environment
    /// pointer, which would put the frame in the wrong register.
    #[test]
    fn the_variant_uses_the_c_convention() {
        let func = counted_loop();
        let v = variant_of(&func);
        assert_eq!(
            v.function.signature.calling_convention,
            crate::ir::CallingConvention::C
        );
    }

    #[test]
    fn the_entry_block_is_refused_with_a_reason() {
        let func = counted_loop();
        let entry = func.cfg.entry_block;
        assert_eq!(
            build_osr_variant(&func, entry, IrFunctionId(9996)).err(),
            Some(OsrReject::IsEntryBlock)
        );
    }

    #[test]
    fn a_probe_resolves_one_register_per_frame_slot() {
        let func = counted_loop();
        let v = variant_of(&func);
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
        let ins = v.layout.live_ins_at(&func, latch).expect("resolves");
        assert_eq!(ins.len(), v.layout.params.len());
    }

    /// A carried pointer stays a phi rather than becoming a frozen frame slot:
    /// the latch allocates a replacement each iteration, and freezing it would
    /// leak every allocation after the first and free the wrong one.
    #[test]
    fn a_rotating_pointer_stays_a_phi() {
        let (func, header, latch, p, _p0, p1) = rotation_loop();
        let v = build_osr_variant(&func, header, IrFunctionId(9994)).expect("variant");
        let phi = v.function.cfg.blocks[&header]
            .phi_nodes
            .iter()
            .find(|n| n.dest == p)
            .expect("carried phi survives");
        assert!(
            phi.incoming
                .iter()
                .any(|(pred, val)| *pred == latch && *val == p1),
            "latch edge lost: incoming={:?}",
            phi.incoming
        );
        assert!(
            v.layout
                .params
                .contains(&OsrParam::HeaderPhi { phi_dest: p }),
            "the carried pointer travels as a phi slot"
        );
    }

    /// The invariant that makes a variant safe to run: it never reads a
    /// register that neither the prologue nor one of its own instructions
    /// produced. A miss here is state the back edge failed to hand over.
    #[test]
    fn variant_reads_nothing_it_was_not_given() {
        for func in [counted_loop(), nested_loop().0, rotation_loop().0] {
            for header in find_loop_headers(&func) {
                let Ok(v) = build_osr_variant(&func, header, IrFunctionId(9995)) else {
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
                                "{}: block {id:?} phi reads {val:?} from {pred:?}, never provided",
                                v.function.name
                            );
                        }
                    }
                    for inst in &block.instructions {
                        for u in inst.uses() {
                            assert!(
                                available.contains(&u),
                                "{}: block {id:?} reads {u:?}, never provided",
                                v.function.name
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
    }

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

    #[test]
    fn a_frame_places_each_slot_at_its_alignment() {
        let frame = OsrFrame::for_types(&[IrType::I8, IrType::F64, IrType::I32]);
        assert_eq!(frame.offsets, vec![0, 8, 16]);
        assert_eq!(frame.align, 8);
        assert_eq!(frame.size % 8, 0);
    }

    /// Generated code embeds a slot's address as a constant, so the address
    /// has to survive the map growing under it. A slot held inline would move
    /// on rehash and every armed probe would load from freed memory.
    #[test]
    fn a_slot_address_survives_the_map_growing() {
        let f = "slot_stability_probe";
        let first = helper_slot_addr(f, 0);
        for site in 1..512u64 {
            helper_slot_addr(f, site);
        }
        assert_eq!(first, helper_slot_addr(f, 0));
    }

    #[test]
    fn a_published_helper_is_what_the_slot_reads_back() {
        let f = "slot_publish_probe";
        let site = encode_osr_site(3, 7);
        assert!(helper_for(f, site).is_null(), "unarmed sites read null");

        let code = 0xDEAD_BEEFusize as *mut ();
        publish_helper(f, site, code);
        assert_eq!(helper_for(f, site), code);

        // The address codegen baked in must observe the publish.
        let slot = helper_slot_addr(f, site) as *const std::sync::atomic::AtomicU64;
        let seen = unsafe { (*slot).load(std::sync::atomic::Ordering::Acquire) };
        assert_eq!(seen, code as u64);
    }

    /// Two sites in one function, and the same site in two functions, must not
    /// share a slot -- either collision dispatches a loop into another loop.
    #[test]
    fn slots_do_not_collide_across_functions_or_sites() {
        let (a, b) = ("slot_fn_a", "slot_fn_b");
        let (s0, s1) = (encode_osr_site(0, 1), encode_osr_site(1, 1));
        // Stand-ins for compiled code: never dereferenced, only compared.
        let (p1, p2, p3) = (
            std::ptr::without_provenance_mut::<()>(1),
            std::ptr::without_provenance_mut::<()>(2),
            std::ptr::without_provenance_mut::<()>(3),
        );
        publish_helper(a, s0, p1);
        publish_helper(a, s1, p2);
        publish_helper(b, s0, p3);
        assert_eq!(helper_for(a, s0), p1);
        assert_eq!(helper_for(a, s1), p2);
        assert_eq!(helper_for(b, s0), p3);
    }

    #[test]
    fn a_site_key_round_trips_its_ordinal_and_count() {
        let site = encode_osr_site(7, 99);
        assert_eq!(decode_osr_site(site), (7, 99));
    }

    /// The site key a variant reports is the one a probe encodes, or the two
    /// disagree about which resume point they mean.
    #[test]
    fn the_layout_site_key_matches_the_header_ordinal() {
        let func = counted_loop();
        let v = variant_of(&func);
        let (ordinal, count) = decode_osr_site(v.layout.site_key());
        assert_eq!(Some(ordinal), loop_ordinal_of(&func, v.site));
        assert_eq!(count as usize, v.layout.params.len());
    }
}
