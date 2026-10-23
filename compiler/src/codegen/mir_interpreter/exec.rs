//! Running decoded functions.

use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

use crate::ir::AtomicRmwOp;

use super::decode::{
    Block, Call, Edge, FloatCc, FloatOp, FnRef, Function, GlobalSlot, IndirectCall, IntCc, IntOp,
    Intrinsic, Mt, Op, Src, Target, Term, canon, float, float_bits, sext,
};
use super::native::{self, Ret};
use super::{InterpError, Machine, trampoline};

/// Interpreted calls deeper than this stop with a stack overflow.
const MAX_DEPTH: u32 = 10_000;

pub(super) struct Frame {
    regs: Vec<u64>,
    slots: Vec<u64>,
    live: Vec<u8>,
    env: u64,
    sret: u64,
}

/// How a run of a frame ended.
enum Flow {
    Return(u64),
    /// The handler guarding the run's `setjmp` was popped; the frame carries on
    /// in the caller of the region from here.
    Popped {
        block: u32,
        pc: usize,
    },
}

fn fault(message: impl Into<String>) -> InterpError {
    InterpError::RuntimeError(message.into())
}

fn trap(message: &str) -> InterpError {
    InterpError::RuntimeError(message.to_string())
}

/// Call `f` with `args` in its ABI order, hidden parameters first.
pub(super) fn call(m: &Machine, f: &Function, args: &[u64]) -> Result<u64, InterpError> {
    let depth = m.depth.get();
    if depth >= MAX_DEPTH {
        return Err(InterpError::StackOverflow);
    }
    let mut fr = Frame {
        regs: vec![0; f.registers],
        slots: vec![0; f.slot_words],
        live: vec![1; f.live_slots],
        env: 0,
        sret: 0,
    };
    let mut at = 0;
    let mut next = || {
        let v = args.get(at).copied().unwrap_or(0);
        at += 1;
        v
    };
    if f.abi.sret {
        fr.sret = next();
    }
    if f.abi.hidden_env {
        fr.env = next();
    }
    for &(reg, mt) in &f.params {
        fr.regs[reg as usize] = canon(next(), mt);
    }
    if f.explicit_env {
        if let Some(&(reg, _)) = f.params.first() {
            fr.env = fr.regs[reg as usize];
        }
    }
    m.depth.set(depth + 1);
    m.tick(f)?;
    let result = run(m, f, &mut fr, f.entry, 0, false);
    m.depth.set(depth);
    match result? {
        Flow::Return(v) => Ok(v),
        Flow::Popped { .. } => Err(fault("exception handler popped outside its try")),
    }
}

#[inline(always)]
fn read(addr: u64, mt: Mt) -> Result<u64, InterpError> {
    if addr < 4096 {
        return Err(fault(format!("read through a null pointer ({:#x})", addr)));
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
        return Err(fault(format!("write through a null pointer ({:#x})", addr)));
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
        return Err(fault("atomic access through a null pointer"));
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
        return Err(fault("atomic access through a null pointer"));
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

/// Run `f` from `pc` in `block` until it returns or, in a `setjmp` region,
/// until the region's handler is popped.
///
/// Every interpreted call nests a `run`, so the common operations are handled
/// here and the rest in `cold`, which keeps this frame small.
fn run(
    m: &Machine,
    f: &Function,
    fr: &mut Frame,
    mut block: u32,
    mut pc: usize,
    in_region: bool,
) -> Result<Flow, InterpError> {
    // Landing pads of nested trys that caught here and have yet to pop.
    let mut caught = 0u32;
    'frame: loop {
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
                Op::Call(c) => call_op(m, fr, c)?,
                Op::CallIndirect(c) => call_indirect(m, fr, c)?,
                Op::SetJmp { dest, buf } => match set_jmp(m, f, fr, block, pc, *dest, *buf)? {
                    Jmp::Finished(flow) => return Ok(flow),
                    Jmp::Resume { block: b, pc: p } => {
                        block = b;
                        pc = p;
                        continue 'frame;
                    }
                    Jmp::Caught => caught += 1,
                },
                Op::HandlerPopped => {
                    if caught > 0 {
                        caught -= 1;
                    } else if in_region {
                        return Ok(Flow::Popped { block, pc: pc + 1 });
                    }
                }
                Op::Bail => {
                    return Err(if m.can_bail() {
                        InterpError::JitBailout(f.id)
                    } else {
                        fault("vector code reached the interpreter inside a native call")
                    });
                }
                Op::Trap { message } => return Err(fault(message.to_string())),
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
                return Ok(Flow::Return(v.map(|r| fr.regs[r as usize]).unwrap_or(0)));
            }
            Term::ReturnSret { value, fields } => {
                let src = fr.regs[*value as usize];
                for &(offset, mt) in fields {
                    let o = offset as i64 as u64;
                    let v = read(src.wrapping_add(o), mt)?;
                    write(fr.sret.wrapping_add(o), mt, v)?;
                }
                return Ok(Flow::Return(fr.sret));
            }
            Term::Trap(message) => return Err(fault(message.to_string())),
        };
        pc = 0;
        m.tick(f)?;
    }
}

/// How a `setjmp` left the frame.
enum Jmp {
    /// The rest of the frame ran inside the region and finished.
    Finished(Flow),
    /// The try body ended; carry on from here.
    Resume { block: u32, pc: usize },
    /// A throw landed: carry on after the `setjmp`, which returned nonzero.
    Caught,
}

#[inline(never)]
fn set_jmp(
    m: &Machine,
    f: &Function,
    fr: &mut Frame,
    block: u32,
    pc: usize,
    dest: u32,
    buf: u32,
) -> Result<Jmp, InterpError> {
    let buf_ptr = fr.regs[buf as usize] as *mut u8;
    let dest = dest as usize;
    let (depth, native_depth) = (m.depth.get(), m.native_depth.get());
    let current = super::current();
    let frame: *mut Frame = fr;
    let mut outcome = None;
    // A bailout cannot leave through the setjmp frame.
    m.native_depth.set(native_depth + 1);
    let delivered = unsafe {
        native::setjmp_region(buf_ptr, &mut || {
            let fr = &mut *frame;
            fr.regs[dest] = 0;
            outcome = Some(run(m, f, fr, block, pc + 1, true));
        })
    };
    // A longjmp skipped whatever restored these.
    m.native_depth.set(native_depth);
    m.depth.set(depth);
    super::set_current(current);
    if delivered != 0 {
        fr.regs[dest] = canon(delivered as u64, Mt::I32);
        return Ok(Jmp::Caught);
    }
    match outcome {
        Some(Ok(Flow::Popped { block, pc })) => Ok(Jmp::Resume { block, pc }),
        Some(Ok(flow)) => Ok(Jmp::Finished(flow)),
        Some(Err(e)) => Err(e),
        None => Err(fault("setjmp region ended without running")),
    }
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

fn src_value(fr: &Frame, s: Src) -> u64 {
    match s {
        Src::Reg(r) => fr.regs[r as usize],
        Src::Imm(v) => v,
        Src::Slot(offset) => fr.slots.as_ptr() as u64 + offset as u64,
    }
}

fn call_op(m: &Machine, fr: &mut Frame, c: &Call) -> Result<(), InterpError> {
    let mut args = smallvec::SmallVec::<[(u64, Mt); 8]>::with_capacity(c.args.len());
    for &(s, mt) in &c.args {
        args.push((canon(src_value(fr, s), mt), mt));
    }
    let value = match &c.target {
        Target::Interp { module, func } => {
            let callee = m.function_at(*module, *func)?;
            let bits: smallvec::SmallVec<[u64; 8]> = args.iter().map(|a| a.0).collect();
            call(m, &callee, &bits)?
        }
        Target::Native(addr) => {
            let ret = match (c.sret, c.ret) {
                (Some((offset, size)), _) => Ret::Sret {
                    dest: fr.slots.as_ptr() as usize + offset as usize,
                    size: size as usize,
                },
                (None, Some(mt)) => Ret::Value(mt),
                (None, None) => Ret::Void,
            };
            m.native(|| unsafe { native::call(*addr, &args, ret) })?
        }
    };
    if let Some(d) = c.dest {
        fr.regs[d as usize] = match (c.sret, c.ret) {
            (Some((offset, _)), _) => fr.slots.as_ptr() as u64 + offset as u64,
            (None, Some(mt)) => canon(value, mt),
            (None, None) => 0,
        };
    }
    Ok(())
}

fn call_indirect(m: &Machine, fr: &mut Frame, c: &IndirectCall) -> Result<(), InterpError> {
    let closure = fr.regs[c.closure as usize];
    if closure == 0 {
        return Err(fault("call through a null function value"));
    }
    let code = read(closure, Mt::I64)?;
    let env = read(closure + 8, Mt::I64)?;
    let mut args = smallvec::SmallVec::<[(u64, Mt); 8]>::with_capacity(c.args.len() + 1);
    args.push((env, Mt::I64));
    for &(r, mt) in &c.args {
        args.push((fr.regs[r as usize], mt));
    }
    let interpreted = trampoline::target_of(code as usize)
        .filter(|t| t.forward.load(std::sync::atomic::Ordering::Acquire) == 0);
    let value = match interpreted {
        Some(t) => {
            let callee = m.function_at(t.module, t.func)?;
            let bits: smallvec::SmallVec<[u64; 8]> = args.iter().map(|a| a.0).collect();
            call(m, &callee, &bits)?
        }
        None => {
            let ret = c.ret.map(Ret::Value).unwrap_or(Ret::Void);
            m.native(|| unsafe { native::call(code as usize, &args, ret) })?
        }
    };
    if let Some(d) = c.dest {
        fr.regs[d as usize] = c.ret.map(|mt| canon(value, mt)).unwrap_or(0);
    }
    Ok(())
}
