//! Running decoded functions.
//!
//! Interpreted calls push frames on an explicit stack rather than nesting Rust
//! calls, so a run can stop with every interpreted frame intact: when a loop
//! gets hot, the stack is set aside while the program is compiled, and the
//! loop then resumes in compiled code while its callers carry on here. Only a
//! native callback or a `setjmp` region starts a nested run.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

use crate::ir::{AtomicRmwOp, IrBlockId, IrFunctionId};

use super::decode::{
    Block, Call, Edge, FloatCc, FloatOp, FnRef, Function, GlobalSlot, IndirectCall, IntCc, IntOp,
    Intrinsic, Mt, Op, Reg, Src, Target, Term, canon, float, float_bits, sext,
};
use super::native::{self, Ret};
use super::{InterpError, Machine, trampoline};

/// Interpreted frames deeper than this stop with a stack overflow.
const MAX_DEPTH: usize = 200_000;

pub(super) struct Frame {
    f: Arc<Function>,
    regs: Vec<u64>,
    slots: Vec<u64>,
    live: Vec<u8>,
    env: u64,
    sret: u64,
    block: u32,
    pc: usize,
    /// Landing pads of trys that caught in this frame and have yet to pop.
    caught: u32,
    /// Where the result of the call this frame waits on goes.
    pending: Pending,
}

#[derive(Clone, Copy, Default)]
struct Pending {
    dest: Option<Reg>,
    ret: Option<Mt>,
    sret: Option<u32>,
}

impl Frame {
    /// A frame for `f`, given its arguments in ABI order, hidden ones first.
    fn new(f: Arc<Function>, args: &[u64]) -> Self {
        let mut fr = Frame {
            regs: vec![0; f.registers],
            slots: vec![0; f.slot_words],
            live: vec![1; f.live_slots],
            env: 0,
            sret: 0,
            block: f.entry,
            pc: 0,
            caught: 0,
            pending: Pending::default(),
            f,
        };
        let mut at = 0;
        let mut next = || {
            let v = args.get(at).copied().unwrap_or(0);
            at += 1;
            v
        };
        if fr.f.abi.sret {
            fr.sret = next();
        }
        if fr.f.abi.hidden_env {
            fr.env = next();
        }
        for i in 0..fr.f.params.len() {
            let (reg, mt) = fr.f.params[i];
            fr.regs[reg as usize] = canon(next(), mt);
        }
        if fr.f.explicit_env {
            if let Some(&(reg, _)) = fr.f.params.first() {
                fr.env = fr.regs[reg as usize];
            }
        }
        fr
    }

    pub(super) fn function(&self) -> &Arc<Function> {
        &self.f
    }

    /// The block the frame is in, and whether it stopped on entering it.
    pub(super) fn position(&self) -> (IrBlockId, bool) {
        (self.f.block_ids[self.block as usize], self.pc == 0)
    }

    /// Whether the frame stopped at an instruction only compiled code runs.
    pub(super) fn at_bail(&self) -> bool {
        matches!(
            self.f.blocks[self.block as usize].ops.get(self.pc),
            Some(Op::Bail)
        )
    }

    fn deliver(&mut self, v: u64) {
        let p = std::mem::take(&mut self.pending);
        if let Some(d) = p.dest {
            self.regs[d as usize] = match (p.sret, p.ret) {
                (Some(offset), _) => self.slots.as_ptr() as u64 + offset as u64,
                (None, Some(mt)) => canon(v, mt),
                (None, None) => 0,
            };
        }
    }
}

/// Why a run stopped without a result.
pub(super) enum Stop {
    /// A loop got hot: compile, then resume the set-aside stack.
    Suspend(IrFunctionId),
    Fail(InterpError),
}

impl From<InterpError> for Stop {
    fn from(e: InterpError) -> Self {
        Stop::Fail(e)
    }
}

/// How a run of frames ended.
enum Flow {
    Return(u64),
    /// The handler guarding the run's `setjmp` was popped.
    Popped,
}

/// What `step` stopped for.
enum Action {
    Call {
        callee: Arc<Function>,
        args: smallvec::SmallVec<[u64; 8]>,
        pending: Pending,
    },
    Return(u64),
    SetJmp {
        dest: Reg,
        buf: Reg,
    },
    Popped,
    Suspend,
    /// Carry on in the compiled resume point for the frame's loop.
    Transfer(Arc<OsrTarget>),
}

/// A compiled resume point and how to fill its frame from registers.
#[derive(Debug)]
pub(crate) struct OsrTarget {
    helper: usize,
    size: usize,
    slots: Vec<(Reg, u32, Mt)>,
    ret: Option<Mt>,
}

fn fault(message: impl Into<String>) -> InterpError {
    InterpError::RuntimeError(message.into())
}

/// The runtime's `rayzor_has_handler` and `rayzor_null_access`, once known.
static HAS_HANDLER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static NULL_ACCESS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Remember the runtime entry points a null access needs.
pub(super) fn note_symbol(name: &str, addr: usize) {
    match name {
        "rayzor_has_handler" => HAS_HANDLER.store(addr, Ordering::Relaxed),
        "rayzor_null_access" => NULL_ACCESS.store(addr, Ordering::Relaxed),
        _ => {}
    }
}

/// A null access. Inside a `try` it throws `Null Object Reference`, as the
/// fault does in compiled code; outside one it stops with `message`.
#[cold]
fn null_fault(message: impl Into<String>) -> InterpError {
    let (has, throw) = (
        HAS_HANDLER.load(Ordering::Relaxed),
        NULL_ACCESS.load(Ordering::Relaxed),
    );
    if has != 0 && throw != 0 {
        let has: extern "C" fn() -> bool = unsafe { std::mem::transmute(has) };
        if has() {
            let throw: extern "C" fn() = unsafe { std::mem::transmute(throw) };
            throw();
        }
    }
    fault(message)
}

fn trap(message: &str) -> InterpError {
    InterpError::RuntimeError(message.to_string())
}

/// Call `f` with `args` in its ABI order, hidden parameters first.
pub(super) fn call(m: &Machine, f: Arc<Function>, args: &[u64]) -> Result<u64, InterpError> {
    let mut stack = vec![Frame::new(f, args)];
    finish(m, stack_run(m, &mut stack), stack)
}

/// Continue a stack set aside by a suspension. A frame that stopped on
/// entering a loop header with a resume point carries on there at once.
pub(super) fn resume(m: &Machine, mut stack: Vec<Frame>) -> Result<u64, InterpError> {
    if let Some(top) = stack.last()
        && top.pc == 0
        && let Some(t) = osr_target(m, &top.f, top.block)
    {
        let v = transfer(m, top, &t)?;
        stack.pop();
        match stack.last_mut() {
            Some(caller) => caller.deliver(v),
            None => return Ok(v),
        }
    }
    let r = stack_run(m, &mut stack);
    finish(m, r, stack)
}

fn stack_run(m: &Machine, stack: &mut Vec<Frame>) -> Result<Flow, Stop> {
    drive(m, stack, 0, None)
}

fn finish(m: &Machine, r: Result<Flow, Stop>, stack: Vec<Frame>) -> Result<u64, InterpError> {
    match r {
        Ok(Flow::Return(v)) => Ok(v),
        Ok(Flow::Popped) => Err(fault("exception handler popped outside its try")),
        Err(Stop::Suspend(id)) => {
            m.suspend(stack);
            Err(InterpError::JitBailout(id))
        }
        Err(Stop::Fail(e)) => Err(e),
    }
}

/// Run the stack until the frame at `floor` returns or, in a `setjmp`
/// region, until the handler of the frame at `region` is popped.
fn drive(
    m: &Machine,
    stack: &mut Vec<Frame>,
    floor: usize,
    region: Option<usize>,
) -> Result<Flow, Stop> {
    loop {
        let top = stack.len() - 1;
        let action = step(m, &mut stack[top], region == Some(top))?;
        let returned = match action {
            Action::Call {
                callee,
                args,
                pending,
            } => {
                if stack.len() >= MAX_DEPTH {
                    return Err(InterpError::StackOverflow.into());
                }
                stack[top].pending = pending;
                stack.push(Frame::new(callee, &args));
                continue;
            }
            Action::Return(v) => v,
            Action::Transfer(t) => transfer(m, &stack[top], &t)?,
            Action::SetJmp { dest, buf } => match set_jmp(m, stack, top, dest, buf)? {
                Some(v) => {
                    // The frame returned inside the region, which popped it.
                    if stack.len() == floor {
                        return Ok(Flow::Return(v));
                    }
                    stack.last_mut().unwrap().deliver(v);
                    continue;
                }
                None => continue,
            },
            Action::Popped => return Ok(Flow::Popped),
            Action::Suspend => return Err(Stop::Suspend(stack[top].f.id)),
        };
        stack.pop();
        if stack.len() == floor {
            return Ok(Flow::Return(returned));
        }
        stack.last_mut().unwrap().deliver(returned);
    }
}

/// Run `setjmp` for the frame at `at` with the rest of that frame inside it.
/// `Some(v)` when the frame returned `v` before its handler was popped.
#[inline(never)]
fn set_jmp(
    m: &Machine,
    stack: &mut Vec<Frame>,
    at: usize,
    dest: Reg,
    buf: Reg,
) -> Result<Option<u64>, Stop> {
    let fr = &mut stack[at];
    let buf_ptr = fr.regs[buf as usize] as *mut u8;
    fr.regs[dest as usize] = 0;
    fr.pc += 1;
    let (block, pc, caught) = (fr.block, fr.pc, fr.caught);
    let native_depth = m.native_depth.get();
    let current = super::current();
    let frames: *mut Vec<Frame> = stack;
    let mut outcome = None;
    // A suspension cannot leave through the setjmp frame.
    m.native_depth.set(native_depth + 1);
    let delivered = unsafe {
        native::setjmp_region(buf_ptr, &mut || {
            outcome = Some(drive(m, &mut *frames, at, Some(at)));
        })
    };
    // A longjmp skipped whatever restored these.
    m.native_depth.set(native_depth);
    super::set_current(current);
    if delivered != 0 {
        // The throw's frames are gone; the try's frame lands after its setjmp.
        stack.truncate(at + 1);
        let fr = &mut stack[at];
        fr.block = block;
        fr.pc = pc;
        fr.caught = caught + 1;
        fr.pending = Pending::default();
        fr.regs[dest as usize] = canon(delivered as u64, Mt::I32);
        return Ok(None);
    }
    match outcome {
        Some(Ok(Flow::Popped)) => Ok(None),
        Some(Ok(Flow::Return(v))) => Ok(Some(v)),
        Some(Err(e)) => Err(e),
        None => Err(fault("setjmp region ended without running").into()),
    }
}

/// Run the compiled resume point with the frame's live values.
fn transfer(m: &Machine, fr: &Frame, t: &OsrTarget) -> Result<u64, InterpError> {
    let mut frame = vec![0u64; t.size.div_ceil(8).max(1)];
    let base = frame.as_mut_ptr() as *mut u8;
    for &(reg, offset, mt) in &t.slots {
        let bytes = fr.regs[reg as usize].to_le_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(offset as usize), mt.bytes())
        };
    }
    let ret = t.ret.map(Ret::Value).unwrap_or(Ret::Void);
    let args = [(base as u64, Mt::I64)];
    m.program.transfers.fetch_add(1, Ordering::Relaxed);
    m.native(|| unsafe { native::call(t.helper, &args, ret) })
}

/// The compiled resume point for `block`, if it is a loop header that has one.
fn osr_target(m: &Machine, f: &Function, block: u32) -> Option<Arc<OsrTarget>> {
    f.osr
        .get_or_init(|| resume_points(m, f))
        .get(&block)
        .cloned()
}

fn resume_points(m: &Machine, f: &Function) -> HashMap<u32, Arc<OsrTarget>> {
    use crate::ir::osr;
    let mut out = HashMap::new();
    for (dense, &header) in f.block_ids.iter().enumerate() {
        let Some((helper, layout)) = m.program.resume_point(f.module, f.id, header) else {
            continue;
        };
        let mut slots = Vec::with_capacity(layout.params.len());
        let mut usable = true;
        for (i, p) in layout.params.iter().enumerate() {
            let reg = match p {
                osr::OsrParam::HeaderPhi { phi_dest } => *phi_dest,
                osr::OsrParam::Live(v) => *v,
            }
            .as_u32();
            // The resume point loads each value at its MIR type; one held at
            // another width here cannot be handed over as it stands.
            let want = Mt::of(&layout.live_in_types[i]);
            let have = f.reg_mts.get(reg as usize).copied();
            if want.is_none() || have != want {
                if osr::osr_trace_enabled() {
                    eprintln!(
                        "[osr] interpreted {} {header:?}: ${reg} is {have:?}, resume point loads {want:?}",
                        f.name
                    );
                }
                usable = false;
                break;
            }
            slots.push((reg, layout.frame.offsets[i], want.unwrap()));
        }
        if usable {
            out.insert(
                dense as u32,
                Arc::new(OsrTarget {
                    helper,
                    size: layout.frame.size as usize,
                    slots,
                    ret: Mt::of(&f.return_type),
                }),
            );
        }
    }
    out
}

/// Count a block entered. Past the threshold, suspend so the program can be
/// compiled; once it has been, a loop header with a resume point transfers.
#[inline(always)]
fn entered(m: &Machine, f: &Function, block: u32) -> Option<Action> {
    if m.program.has_natives() {
        return osr_target(m, f, block).map(Action::Transfer);
    }
    let n = m.iterations.get() + 1;
    m.iterations.set(n);
    if n >= m.max_iterations.get() && m.can_bail() {
        m.iterations.set(0);
        return Some(Action::Suspend);
    }
    None
}

#[inline(always)]
fn read(addr: u64, mt: Mt) -> Result<u64, InterpError> {
    if addr < 4096 {
        return Err(null_fault(format!(
            "read through a null pointer ({:#x})",
            addr
        )));
    }
    let p = addr as *const u8;
    Ok(unsafe {
        match mt {
            Mt::I8 => p.read() as u64,
            Mt::I16 => (p as *const u16).read_unaligned() as u64,
            Mt::I32 | Mt::F32 => (p as *const u32).read_unaligned() as u64,
            Mt::I64 | Mt::F64 => (p as *const u64).read_unaligned(),
            Mt::V128 => return Err(fault("vector load in the interpreter")),
        }
    })
}

#[inline(always)]
fn write(addr: u64, mt: Mt, bits: u64) -> Result<(), InterpError> {
    if addr < 4096 {
        return Err(null_fault(format!(
            "write through a null pointer ({:#x})",
            addr
        )));
    }
    let p = addr as *mut u8;
    unsafe {
        match mt {
            Mt::I8 => p.write(bits as u8),
            Mt::I16 => (p as *mut u16).write_unaligned(bits as u16),
            Mt::I32 | Mt::F32 => (p as *mut u32).write_unaligned(bits as u32),
            Mt::I64 | Mt::F64 => (p as *mut u64).write_unaligned(bits),
            Mt::V128 => return Err(fault("vector store in the interpreter")),
        }
    }
    Ok(())
}

fn malloc(size: usize) -> Result<u64, InterpError> {
    let p = unsafe { libc::malloc(size) };
    if p.is_null() {
        return Err(fault("out of memory"));
    }
    Ok(p as u64)
}

#[inline(always)]
fn ibin(op: IntOp, mt: Mt, a: u64, b: u64) -> Result<u64, InterpError> {
    let bits = mt.bits();
    Ok(canon(
        match op {
            IntOp::Add => a.wrapping_add(b),
            IntOp::Sub => a.wrapping_sub(b),
            IntOp::Mul => a.wrapping_mul(b),
            IntOp::SDiv => {
                let (x, y) = (sext(a, mt), sext(b, mt));
                if y == 0 {
                    return Err(trap("integer division by zero"));
                }
                if y == -1 && x == sext(1u64 << (bits - 1), mt) {
                    return Err(trap("integer overflow"));
                }
                (x / y) as u64
            }
            IntOp::UDiv => {
                if b == 0 {
                    return Err(trap("integer division by zero"));
                }
                a / b
            }
            IntOp::SRem => {
                let (x, y) = (sext(a, mt), sext(b, mt));
                if y == 0 {
                    return Err(trap("integer division by zero"));
                }
                x.wrapping_rem(y) as u64
            }
            IntOp::URem => {
                if b == 0 {
                    return Err(trap("integer division by zero"));
                }
                a % b
            }
            IntOp::And => a & b,
            IntOp::Or => a | b,
            IntOp::Xor => a ^ b,
            IntOp::Shl => a << (b & (bits as u64 - 1)),
            IntOp::SShr => (sext(a, mt) >> (b & (bits as u64 - 1))) as u64,
            IntOp::UShr => a >> (b & (bits as u64 - 1)),
        },
        mt,
    ))
}

#[inline(always)]
fn fbin(op: FloatOp, mt: Mt, a: u64, b: u64) -> u64 {
    if mt == Mt::F32 {
        let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
        let r = match op {
            FloatOp::Add => x + y,
            FloatOp::Sub => x - y,
            FloatOp::Mul => x * y,
            FloatOp::Div => x / y,
            FloatOp::Rem => x - (x / y).trunc() * y,
        };
        r.to_bits() as u64
    } else {
        let (x, y) = (f64::from_bits(a), f64::from_bits(b));
        let r = match op {
            FloatOp::Add => x + y,
            FloatOp::Sub => x - y,
            FloatOp::Mul => x * y,
            FloatOp::Div => x / y,
            FloatOp::Rem => x - (x / y).trunc() * y,
        };
        r.to_bits()
    }
}

fn icmp(cc: IntCc, mt: Mt, a: u64, b: u64) -> bool {
    let (x, y) = (sext(a, mt), sext(b, mt));
    let (u, v) = (canon(a, mt), canon(b, mt));
    match cc {
        IntCc::Eq => u == v,
        IntCc::Ne => u != v,
        IntCc::Slt => x < y,
        IntCc::Sle => x <= y,
        IntCc::Sgt => x > y,
        IntCc::Sge => x >= y,
        IntCc::Ult => u < v,
        IntCc::Ule => u <= v,
        IntCc::Ugt => u > v,
        IntCc::Uge => u >= v,
    }
}

fn fcmp(cc: FloatCc, mt: Mt, a: u64, b: u64) -> bool {
    let (x, y) = (float(a, mt), float(b, mt));
    match cc {
        FloatCc::Eq => x == y,
        FloatCc::Ne => x != y,
        FloatCc::Lt => x < y,
        FloatCc::Le => x <= y,
        FloatCc::Gt => x > y,
        FloatCc::Ge => x >= y,
        FloatCc::Ord => !x.is_nan() && !y.is_nan(),
        FloatCc::Uno => x.is_nan() || y.is_nan(),
    }
}

fn math(kind: Intrinsic, mt: Mt, bits: u64) -> u64 {
    if mt == Mt::F32 {
        let x = f32::from_bits(bits as u32);
        let r = match kind {
            Intrinsic::Sqrt => return x.sqrt().to_bits() as u64,
            Intrinsic::Abs => return x.abs().to_bits() as u64,
            Intrinsic::Floor => x.floor(),
            Intrinsic::Ceil => x.ceil(),
            Intrinsic::Round => {
                let down = x.floor();
                if x - down >= 0.5 { down + 1.0 } else { down }
            }
        };
        canon(r as i32 as u64, Mt::I32)
    } else {
        let x = f64::from_bits(bits);
        let r = match kind {
            Intrinsic::Sqrt => return x.sqrt().to_bits(),
            Intrinsic::Abs => return x.abs().to_bits(),
            Intrinsic::Floor => x.floor(),
            Intrinsic::Ceil => x.ceil(),
            Intrinsic::Round => {
                let down = x.floor();
                if x - down >= 0.5 { down + 1.0 } else { down }
            }
        };
        canon(r as i32 as u64, Mt::I32)
    }
}

fn atomic_rmw(mt: Mt, op: AtomicRmwOp, ptr: u64, val: u64) -> Result<u64, InterpError> {
    if ptr < 4096 {
        return Err(null_fault("atomic access through a null pointer"));
    }
    macro_rules! rmw {
        ($a:ty, $t:ty) => {{
            let a = unsafe { <$a>::from_ptr(ptr as *mut $t) };
            let v = val as $t;
            let o = Ordering::SeqCst;
            (match op {
                AtomicRmwOp::Add => a.fetch_add(v, o),
                AtomicRmwOp::Sub => a.fetch_sub(v, o),
                AtomicRmwOp::And => a.fetch_and(v, o),
                AtomicRmwOp::Or => a.fetch_or(v, o),
                AtomicRmwOp::Xor => a.fetch_xor(v, o),
                AtomicRmwOp::Xchg => a.swap(v, o),
            }) as u64
        }};
    }
    Ok(match mt {
        Mt::I8 => rmw!(AtomicU8, u8),
        Mt::I16 => rmw!(AtomicU16, u16),
        Mt::I32 | Mt::F32 => rmw!(AtomicU32, u32),
        _ => rmw!(AtomicU64, u64),
    })
}

fn atomic_cas(mt: Mt, ptr: u64, expected: u64, new: u64) -> Result<u64, InterpError> {
    if ptr < 4096 {
        return Err(null_fault("atomic access through a null pointer"));
    }
    macro_rules! cas {
        ($a:ty, $t:ty) => {{
            let a = unsafe { <$a>::from_ptr(ptr as *mut $t) };
            match a.compare_exchange(
                expected as $t,
                new as $t,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(v) | Err(v) => v as u64,
            }
        }};
    }
    Ok(match mt {
        Mt::I8 => cas!(AtomicU8, u8),
        Mt::I16 => cas!(AtomicU16, u16),
        Mt::I32 | Mt::F32 => cas!(AtomicU32, u32),
        _ => cas!(AtomicU64, u64),
    })
}

fn take(fr: &mut Frame, e: &Edge) -> Result<u32, InterpError> {
    match e.moves.len() {
        0 => {}
        1 => {
            let (dest, src, c1, c2) = &e.moves[0];
            let v = convert(fr.regs[*src as usize], *c1, *c2)?;
            fr.regs[*dest as usize] = v;
        }
        _ => {
            let mut values = smallvec::SmallVec::<[u64; 8]>::with_capacity(e.moves.len());
            for (_, src, c1, c2) in &e.moves {
                values.push(convert(fr.regs[*src as usize], *c1, *c2)?);
            }
            for ((dest, ..), v) in e.moves.iter().zip(values) {
                fr.regs[*dest as usize] = v;
            }
        }
    }
    Ok(e.block)
}

#[inline(always)]
fn convert(
    mut v: u64,
    c1: Option<super::decode::Conv>,
    c2: Option<super::decode::Conv>,
) -> Result<u64, InterpError> {
    if let Some(c) = c1 {
        v = c.apply(v).map_err(trap)?;
    }
    if let Some(c) = c2 {
        v = c.apply(v).map_err(trap)?;
    }
    Ok(v)
}

/// Run the frame from where it stands until it calls, returns, or has to stop.
/// `region_here` says the drive running it ends when this frame pops a handler.
///
/// The common operations are handled here and the rest in `cold`.
fn step(m: &Machine, fr: &mut Frame, region_here: bool) -> Result<Action, Stop> {
    // SAFETY: the frame holds the Arc, and this run does not outlive it.
    let f: &Function = unsafe { &*Arc::as_ptr(&fr.f) };
    let mut block = fr.block;
    let mut pc = fr.pc;
    if pc == 0 && m.program.has_natives() {
        if let Some(t) = osr_target(m, f, block) {
            return Ok(Action::Transfer(t));
        }
    }
    loop {
        let b: &Block = &f.blocks[block as usize];
        while pc < b.ops.len() {
            let r = &mut fr.regs;
            match &b.ops[pc] {
                Op::Const { dest, bits } => r[*dest as usize] = *bits,
                Op::Copy { dest, src } => r[*dest as usize] = r[*src as usize],
                Op::Conv { dest, src, conv } => {
                    r[*dest as usize] = conv.apply(r[*src as usize]).map_err(trap)?;
                }
                Op::IBin { dest, op, mt, a, b } => {
                    r[*dest as usize] = ibin(*op, *mt, r[*a as usize], r[*b as usize])?;
                }
                Op::FBin { dest, op, mt, a, b } => {
                    r[*dest as usize] = fbin(*op, *mt, r[*a as usize], r[*b as usize]);
                }
                Op::ICmp { dest, cc, mt, a, b } => {
                    r[*dest as usize] = icmp(*cc, *mt, r[*a as usize], r[*b as usize]) as u64;
                }
                Op::FCmp { dest, cc, mt, a, b } => {
                    r[*dest as usize] = fcmp(*cc, *mt, r[*a as usize], r[*b as usize]) as u64;
                }
                Op::Select { dest, cond, a, b } => {
                    r[*dest as usize] = if r[*cond as usize] != 0 {
                        r[*a as usize]
                    } else {
                        r[*b as usize]
                    };
                }
                Op::Load {
                    dest,
                    mt,
                    ptr,
                    offset,
                } => {
                    let addr = r[*ptr as usize].wrapping_add(*offset as i64 as u64);
                    r[*dest as usize] = read(addr, *mt)?;
                }
                Op::Store {
                    mt,
                    ptr,
                    val,
                    offset,
                } => {
                    let addr = r[*ptr as usize].wrapping_add(*offset as i64 as u64);
                    write(addr, *mt, r[*val as usize])?;
                }
                Op::AddrIndex {
                    dest,
                    base,
                    index,
                    index_mt,
                    scale,
                } => {
                    let i = sext(r[*index as usize], *index_mt);
                    r[*dest as usize] =
                        r[*base as usize].wrapping_add(i.wrapping_mul(*scale) as u64);
                }
                Op::Call(c) => {
                    if let Some(action) = call_op(m, fr, c)? {
                        fr.block = block;
                        fr.pc = pc + 1;
                        return Ok(action);
                    }
                }
                Op::CallIndirect(c) => {
                    if let Some(action) = call_indirect(m, fr, c)? {
                        fr.block = block;
                        fr.pc = pc + 1;
                        return Ok(action);
                    }
                }
                Op::SetJmp { dest, buf } => {
                    fr.block = block;
                    fr.pc = pc;
                    return Ok(Action::SetJmp {
                        dest: *dest,
                        buf: *buf,
                    });
                }
                Op::HandlerPopped => {
                    if fr.caught > 0 {
                        fr.caught -= 1;
                    } else if region_here {
                        fr.block = block;
                        fr.pc = pc + 1;
                        return Ok(Action::Popped);
                    }
                }
                Op::Bail => {
                    if m.can_bail() && !m.program.has_natives() {
                        fr.block = block;
                        fr.pc = pc;
                        return Ok(Action::Suspend);
                    }
                    return Err(fault("vector code reached the interpreter").into());
                }
                Op::Trap { message } => return Err(fault(message.to_string()).into()),
                op => cold(m, fr, op)?,
            }
            pc += 1;
        }

        block = match &b.term {
            Term::Jump(e) => take(fr, e)?,
            Term::Brif {
                cond,
                then,
                otherwise,
            } => {
                if fr.regs[*cond as usize] != 0 {
                    take(fr, then)?
                } else {
                    take(fr, otherwise)?
                }
            }
            Term::Switch {
                value,
                mt,
                cases,
                default,
            } => {
                let v = sext(fr.regs[*value as usize], *mt);
                let e = cases
                    .iter()
                    .find(|(c, _)| *c == v)
                    .map(|(_, e)| e)
                    .unwrap_or(default);
                take(fr, e)?
            }
            Term::Return(v) => {
                return Ok(Action::Return(v.map(|r| fr.regs[r as usize]).unwrap_or(0)));
            }
            Term::ReturnSret { value, fields } => {
                let src = fr.regs[*value as usize];
                for &(offset, mt) in fields {
                    let o = offset as i64 as u64;
                    let v = read(src.wrapping_add(o), mt)?;
                    write(fr.sret.wrapping_add(o), mt, v)?;
                }
                return Ok(Action::Return(fr.sret));
            }
            Term::Trap(message) => return Err(fault(message.to_string()).into()),
        };
        pc = 0;
        if let Some(action) = entered(m, f, block) {
            fr.block = block;
            fr.pc = 0;
            return Ok(action);
        }
    }
}

fn src_value(fr: &Frame, s: Src) -> u64 {
    match s {
        Src::Reg(r) => fr.regs[r as usize],
        Src::Imm(v) => v,
        Src::Slot(offset) => fr.slots.as_ptr() as u64 + offset as u64,
    }
}

/// A direct call. Native callees run now; an interpreted one comes back as an
/// action for the stack to push.
fn call_op(m: &Machine, fr: &mut Frame, c: &Call) -> Result<Option<Action>, InterpError> {
    let mut args = smallvec::SmallVec::<[(u64, Mt); 8]>::with_capacity(c.args.len());
    for &(s, mt) in &c.args {
        args.push((canon(src_value(fr, s), mt), mt));
    }
    let pending = Pending {
        dest: c.dest,
        ret: c.ret,
        sret: c.sret.map(|(offset, _)| offset),
    };
    let code = match &c.target {
        Target::Native(addr) => *addr,
        Target::Interp { module, func } => match m.program.native(*func) {
            Some(code) => code,
            None => {
                return Ok(Some(Action::Call {
                    callee: m.function_at(*module, *func)?,
                    args: args.iter().map(|a| a.0).collect(),
                    pending,
                }));
            }
        },
    };
    let ret = match (c.sret, c.ret) {
        (Some((offset, size)), _) => Ret::Sret {
            dest: fr.slots.as_ptr() as usize + offset as usize,
            size: size as usize,
        },
        (None, Some(mt)) => Ret::Value(mt),
        (None, None) => Ret::Void,
    };
    let value = m.native(|| unsafe { native::call(code, &args, ret) })?;
    fr.pending = pending;
    fr.deliver(value);
    Ok(None)
}

fn call_indirect(
    m: &Machine,
    fr: &mut Frame,
    c: &IndirectCall,
) -> Result<Option<Action>, InterpError> {
    let closure = fr.regs[c.closure as usize];
    if closure == 0 {
        return Err(null_fault("call through a null function value"));
    }
    let code = read(closure, Mt::I64)?;
    let env = read(closure + 8, Mt::I64)?;
    let mut args = smallvec::SmallVec::<[(u64, Mt); 8]>::with_capacity(c.args.len() + 1);
    args.push((env, Mt::I64));
    for &(r, mt) in &c.args {
        args.push((fr.regs[r as usize], mt));
    }
    let pending = Pending {
        dest: c.dest,
        ret: c.ret,
        sret: None,
    };
    let interpreted =
        trampoline::target_of(code as usize).filter(|t| t.forward.load(Ordering::Acquire) == 0);
    if let Some(t) = interpreted {
        return Ok(Some(Action::Call {
            callee: m.function_at(t.module, t.func)?,
            args: args.iter().map(|a| a.0).collect(),
            pending,
        }));
    }
    let ret = c.ret.map(Ret::Value).unwrap_or(Ret::Void);
    let value = m.native(|| unsafe { native::call(code as usize, &args, ret) })?;
    fr.pending = pending;
    fr.deliver(value);
    Ok(None)
}

/// The operations that are rare in hot code.
#[inline(never)]
fn cold(m: &Machine, fr: &mut Frame, op: &Op) -> Result<(), InterpError> {
    let r = &mut fr.regs;
    match op {
        Op::Str {
            dest,
            literal,
            ptr,
            len,
        } => {
            let make: extern "C" fn(usize, usize) -> u64 = unsafe { std::mem::transmute(*literal) };
            r[*dest as usize] = make(*ptr, *len);
        }
        Op::Fma {
            dest,
            mt,
            a,
            b,
            c,
            neg_a,
            neg_c,
        } => {
            let (x, y, z) = (
                float(r[*a as usize], *mt),
                float(r[*b as usize], *mt),
                float(r[*c as usize], *mt),
            );
            let x = if *neg_a { -x } else { x };
            let z = if *neg_c { -z } else { z };
            r[*dest as usize] = if *mt == Mt::F32 {
                (x as f32).mul_add(y as f32, z as f32).to_bits() as u64
            } else {
                x.mul_add(y, z).to_bits()
            };
        }
        Op::INeg { dest, mt, a } => {
            r[*dest as usize] = canon(0u64.wrapping_sub(r[*a as usize]), *mt);
        }
        Op::FNeg { dest, mt, a } => {
            let sign = if *mt == Mt::F32 {
                1u64 << 31
            } else {
                1u64 << 63
            };
            r[*dest as usize] = r[*a as usize] ^ sign;
        }
        Op::BNot { dest, mt, a } => r[*dest as usize] = canon(!r[*a as usize], *mt),
        Op::IsZero { dest, a } => r[*dest as usize] = (r[*a as usize] == 0) as u64,
        Op::SlotAddr { dest, offset } => {
            r[*dest as usize] = fr.slots.as_ptr() as u64 + *offset as u64;
        }
        Op::ArrayHeader {
            dest,
            array,
            offset,
            guarded,
        } => {
            let a = r[*array as usize];
            r[*dest as usize] = if a == 0 && *guarded {
                0
            } else if a == 0 {
                return Err(fault("array header read of a null array"));
            } else {
                read(a + *offset as u64, Mt::I64)?
            };
        }
        Op::ArrayGetPtr {
            dest,
            array,
            index,
            index_mt,
        } => {
            let a = r[*array as usize];
            let data = read(a, Mt::I64)?;
            let elem = read(a + 24, Mt::I64)?;
            let i = sext(r[*index as usize], *index_mt) as u64;
            r[*dest as usize] = data.wrapping_add(i.wrapping_mul(elem));
        }
        Op::Math {
            dest,
            kind,
            arg,
            mt,
        } => r[*dest as usize] = math(*kind, *mt, r[*arg as usize]),
        Op::GlobalLoad { dest, mt, slot } => {
            let word = match *slot {
                GlobalSlot::Address(a) => unsafe { (a as *const u64).read() },
                GlobalSlot::Runtime { id, load, .. } => {
                    let load: extern "C" fn(i64) -> u64 = unsafe { std::mem::transmute(load) };
                    load(i64::from(id))
                }
            };
            r[*dest as usize] = canon(word, *mt);
        }
        Op::GlobalStore { src, mt, slot } => {
            let word = canon(r[*src as usize], *mt);
            match *slot {
                GlobalSlot::Address(a) => unsafe { (a as *mut u64).write(word) },
                GlobalSlot::Runtime { id, store, .. } => {
                    let store: extern "C" fn(i64, u64) = unsafe { std::mem::transmute(store) };
                    store(i64::from(id), word);
                }
            }
        }
        Op::FnAddr { dest, module, func } => {
            r[*dest as usize] = m.function_address(*module, *func)? as u64;
        }
        Op::Closure {
            dest,
            func,
            captures,
        } => {
            let code = match func {
                FnRef::Address(a) => *a as u64,
                FnRef::Function { module, func } => m.function_address(*module, *func)? as u64,
            };
            let env = if captures.is_empty() {
                0
            } else {
                let env = malloc(captures.len() * 8)?;
                for (i, (reg, mt)) in captures.iter().enumerate() {
                    write(env + 8 * i as u64, *mt, r[*reg as usize])?;
                }
                env
            };
            let obj = malloc(16)?;
            write(obj, Mt::I64, code)?;
            write(obj + 8, Mt::I64, env)?;
            r[*dest as usize] = obj;
        }
        Op::Env { dest } => r[*dest as usize] = fr.env,
        Op::CloneObject { dest, src } => {
            let s = r[*src as usize];
            if s == 0 {
                return Err(fault("clone of a null object"));
            }
            let p = malloc(64)?;
            unsafe { std::ptr::copy_nonoverlapping(s as *const u8, p as *mut u8, 64) };
            r[*dest as usize] = p;
        }
        Op::Free { ptr } => unsafe { libc::free(r[*ptr as usize] as *mut libc::c_void) },
        Op::MemCopy {
            dest,
            src,
            size,
            size_mt,
        } => {
            let n = sext(r[*size as usize], *size_mt).max(0) as usize;
            if n > 0 {
                unsafe {
                    std::ptr::copy(
                        r[*src as usize] as *const u8,
                        r[*dest as usize] as *mut u8,
                        n,
                    )
                };
            }
        }
        Op::MemSet {
            dest,
            val,
            size,
            size_mt,
        } => {
            let n = sext(r[*size as usize], *size_mt).max(0) as usize;
            if n > 0 {
                unsafe {
                    std::ptr::write_bytes(r[*dest as usize] as *mut u8, r[*val as usize] as u8, n)
                };
            }
        }
        Op::AtomicLoad { dest, mt, ptr } => {
            r[*dest as usize] = atomic_rmw(*mt, AtomicRmwOp::Or, r[*ptr as usize], 0)?;
        }
        Op::AtomicStore { mt, ptr, val } => {
            atomic_rmw(*mt, AtomicRmwOp::Xchg, r[*ptr as usize], r[*val as usize])?;
        }
        Op::AtomicRmw {
            dest,
            mt,
            op,
            ptr,
            val,
        } => r[*dest as usize] = atomic_rmw(*mt, *op, r[*ptr as usize], r[*val as usize])?,
        Op::AtomicCas {
            dest,
            mt,
            ptr,
            expected,
            replacement,
        } => {
            r[*dest as usize] = atomic_cas(
                *mt,
                r[*ptr as usize],
                r[*expected as usize],
                r[*replacement as usize],
            )?;
        }
        Op::MarkMoved { slot } => fr.live[*slot as usize] = 0,
        Op::CheckLive { slot, name, line } => {
            if fr.live[*slot as usize] == 0 {
                if let Some(panic) = m.program.symbol("rayzor_panic_use_after_move") {
                    let panic: extern "C" fn(*const u8, i64, i64) =
                        unsafe { std::mem::transmute(panic) };
                    panic(name.as_ptr(), name.len() as i64, i64::from(*line));
                }
                return Err(fault(format!("use of `{}` after move", name)));
            }
        }
        // The hot operations and control transfers are `run`'s.
        _ => unreachable!("{:?} reached the cold path", op),
    }
    Ok(())
}
