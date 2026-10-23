//! Calls from the interpreter into native code, under the platform C ABI.
//!
//! A callee is called through one function type per return class that declares
//! every argument register plus a run of stack words. Registers and stack
//! words the callee does not read are harmless; the ones it does read hold its
//! arguments, placed by the same rules its compiler used.

use super::decode::{Mt, canon};

/// Stack argument words passed on every call.
const STACK_WORDS: usize = 16;

/// The largest struct a native callee may return through memory.
const SRET_WORDS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct Sret([u64; SRET_WORDS]);

#[derive(Clone, Copy)]
pub(super) enum Ret {
    Void,
    Value(Mt),
    /// Returned through memory: the callee's struct is copied to `dest`.
    Sret {
        dest: usize,
        size: usize,
    },
}

/// Arguments assigned to registers and stack words.
struct Placed {
    ints: [u64; 8],
    floats: [f64; 8],
    stack: [u64; STACK_WORDS],
    /// Win64 assigns the first four by position: whether each is a float.
    #[allow(dead_code)]
    float_at: [bool; 4],
}

impl Placed {
    fn new() -> Self {
        Self {
            ints: [0; 8],
            floats: [0.0; 8],
            stack: [0; STACK_WORDS],
            float_at: [false; 4],
        }
    }
}

/// Integer argument registers.
#[cfg(all(target_arch = "x86_64", not(windows)))]
const INT_REGS: usize = 6;
#[cfg(target_arch = "aarch64")]
const INT_REGS: usize = 8;

/// Appends one stack argument. Apple's arm64 ABI packs stack arguments at
/// their natural size and alignment; the others give each an 8-byte slot.
struct StackWriter {
    bytes: [u8; STACK_WORDS * 8],
    len: usize,
}

impl StackWriter {
    fn new() -> Self {
        Self {
            bytes: [0; STACK_WORDS * 8],
            len: 0,
        }
    }

    fn push(&mut self, bits: u64, mt: Mt) -> Result<(), String> {
        let size = if cfg!(all(target_arch = "aarch64", target_vendor = "apple")) {
            mt.bytes().max(1)
        } else {
            8
        };
        let at = self.len.next_multiple_of(size);
        if at + size > self.bytes.len() {
            return Err(format!(
                "native call needs more than {} stack argument words",
                STACK_WORDS
            ));
        }
        self.bytes[at..at + size].copy_from_slice(&bits.to_le_bytes()[..size]);
        self.len = at + size;
        Ok(())
    }

    fn words(&self) -> [u64; STACK_WORDS] {
        let mut out = [0u64; STACK_WORDS];
        for (i, w) in out.iter_mut().enumerate() {
            *w = u64::from_le_bytes(self.bytes[i * 8..i * 8 + 8].try_into().unwrap());
        }
        out
    }
}

fn place(args: &[(u64, Mt)]) -> Result<Placed, String> {
    let mut placed = Placed::new();
    let mut stack = StackWriter::new();
    #[cfg(not(all(target_arch = "x86_64", windows)))]
    {
        let (mut ni, mut nf) = (0, 0);
        for &(bits, mt) in args {
            if mt == Mt::V128 {
                return Err("native call with a vector argument".to_string());
            }
            if mt.is_float() {
                if nf < 8 {
                    placed.floats[nf] = f64::from_bits(bits);
                    nf += 1;
                    continue;
                }
            } else if ni < INT_REGS {
                placed.ints[ni] = bits;
                ni += 1;
                continue;
            }
            stack.push(bits, mt)?;
        }
    }
    #[cfg(all(target_arch = "x86_64", windows))]
    for (i, &(bits, mt)) in args.iter().enumerate() {
        if mt == Mt::V128 {
            return Err("native call with a vector argument".to_string());
        }
        if i < 4 {
            if mt.is_float() {
                placed.floats[i] = f64::from_bits(bits);
                placed.float_at[i] = true;
            } else {
                placed.ints[i] = bits;
            }
        } else {
            stack.push(bits, mt)?;
        }
    }
    placed.stack = stack.words();
    Ok(placed)
}

/// Call `target` with `args`, each the value's register bits and machine type
/// in ABI order. The result comes back as register bits of its type.
///
/// # Safety
/// `target` must be a function with exactly this signature.
pub(super) unsafe fn call(target: usize, args: &[(u64, Mt)], ret: Ret) -> Result<u64, String> {
    let args = match ret {
        // The struct-return pointer is the hidden parameter Rust passes itself.
        Ret::Sret { .. } => &args[1.min(args.len())..],
        _ => args,
    };
    let p = place(args)?;
    unsafe {
        match ret {
            Ret::Void => {
                invoke::<()>(target, &p);
                Ok(0)
            }
            Ret::Value(Mt::F32) => Ok(invoke::<f32>(target, &p).to_bits() as u64),
            Ret::Value(Mt::F64) => Ok(invoke::<f64>(target, &p).to_bits()),
            Ret::Value(mt) => Ok(canon(invoke::<u64>(target, &p), mt)),
            Ret::Sret { dest, size } => {
                if size > SRET_WORDS * 8 {
                    return Err(format!("native struct return of {} bytes", size));
                }
                let out = invoke::<Sret>(target, &p);
                std::ptr::copy_nonoverlapping(out.0.as_ptr() as *const u8, dest as *mut u8, size);
                Ok(dest as u64)
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(windows)))]
unsafe fn invoke<R>(target: usize, p: &Placed) -> R {
    type F<R> = extern "C" fn(
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
    ) -> R;
    let f: F<R> = unsafe { std::mem::transmute(target) };
    let (i, x, s) = (&p.ints, &p.floats, &p.stack);
    f(
        i[0], i[1], i[2], i[3], i[4], i[5], x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], s[0],
        s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13], s[14],
        s[15],
    )
}

#[cfg(target_arch = "aarch64")]
unsafe fn invoke<R>(target: usize, p: &Placed) -> R {
    type F<R> = extern "C" fn(
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
    ) -> R;
    let f: F<R> = unsafe { std::mem::transmute(target) };
    let (i, x, s) = (&p.ints, &p.floats, &p.stack);
    f(
        i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], x[0], x[1], x[2], x[3], x[4], x[5], x[6],
        x[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12],
        s[13], s[14], s[15],
    )
}

/// Win64 gives each of the first four arguments one register, an integer or a
/// vector one by its class, so each class pattern needs its own function type.
#[cfg(all(target_arch = "x86_64", windows))]
unsafe fn invoke<R>(target: usize, p: &Placed) -> R {
    macro_rules! go {
        ($t0:ty, $t1:ty, $t2:ty, $t3:ty, $a0:expr, $a1:expr, $a2:expr, $a3:expr) => {{
            type F<R> = extern "C" fn(
                $t0,
                $t1,
                $t2,
                $t3,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
            ) -> R;
            let f: F<R> = unsafe { std::mem::transmute(target) };
            let s = &p.stack;
            f(
                $a0, $a1, $a2, $a3, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9],
                s[10], s[11], s[12], s[13], s[14], s[15],
            )
        }};
    }
    let (i, x) = (&p.ints, &p.floats);
    match p.float_at {
        [false, false, false, false] => go!(u64, u64, u64, u64, i[0], i[1], i[2], i[3]),
        [true, false, false, false] => go!(f64, u64, u64, u64, x[0], i[1], i[2], i[3]),
        [false, true, false, false] => go!(u64, f64, u64, u64, i[0], x[1], i[2], i[3]),
        [true, true, false, false] => go!(f64, f64, u64, u64, x[0], x[1], i[2], i[3]),
        [false, false, true, false] => go!(u64, u64, f64, u64, i[0], i[1], x[2], i[3]),
        [true, false, true, false] => go!(f64, u64, f64, u64, x[0], i[1], x[2], i[3]),
        [false, true, true, false] => go!(u64, f64, f64, u64, i[0], x[1], x[2], i[3]),
        [true, true, true, false] => go!(f64, f64, f64, u64, x[0], x[1], x[2], i[3]),
        [false, false, false, true] => go!(u64, u64, u64, f64, i[0], i[1], i[2], x[3]),
        [true, false, false, true] => go!(f64, u64, u64, f64, x[0], i[1], i[2], x[3]),
        [false, true, false, true] => go!(u64, f64, u64, f64, i[0], x[1], i[2], x[3]),
        [true, true, false, true] => go!(f64, f64, u64, f64, x[0], x[1], i[2], x[3]),
        [false, false, true, true] => go!(u64, u64, f64, f64, i[0], i[1], x[2], x[3]),
        [true, false, true, true] => go!(f64, u64, f64, f64, x[0], i[1], x[2], x[3]),
        [false, true, true, true] => go!(u64, f64, f64, f64, i[0], x[1], x[2], x[3]),
        [true, true, true, true] => go!(f64, f64, f64, f64, x[0], x[1], x[2], x[3]),
    }
}

/// Read the arguments a native caller passed, as saved by the trampoline
/// entry, back into ABI order.
///
/// # Safety
/// `stack` must point at the caller's stack arguments.
pub(super) unsafe fn unplace(
    params: &[Mt],
    ints: &[u64],
    floats: &[u64],
    stack: *const u8,
) -> Vec<u64> {
    let mut out = Vec::with_capacity(params.len());
    let mut at = 0usize;
    let mut from_stack = |mt: Mt| -> u64 {
        let size = if cfg!(all(target_arch = "aarch64", target_vendor = "apple")) {
            mt.bytes().max(1)
        } else {
            8
        };
        at = at.next_multiple_of(size);
        let mut word = [0u8; 8];
        unsafe { std::ptr::copy_nonoverlapping(stack.add(at), word.as_mut_ptr(), size.min(8)) };
        at += size;
        canon(u64::from_le_bytes(word), mt)
    };
    #[cfg(not(all(target_arch = "x86_64", windows)))]
    {
        let (mut ni, mut nf) = (0, 0);
        for &mt in params {
            let bits = if mt.is_float() {
                if nf < 8 {
                    nf += 1;
                    floats[nf - 1]
                } else {
                    from_stack(mt)
                }
            } else if ni < INT_REGS {
                ni += 1;
                ints[ni - 1]
            } else {
                from_stack(mt)
            };
            out.push(canon(bits, mt));
        }
    }
    #[cfg(all(target_arch = "x86_64", windows))]
    for (i, &mt) in params.iter().enumerate() {
        let bits = if i < 4 {
            if mt.is_float() { floats[i] } else { ints[i] }
        } else {
            from_stack(mt)
        };
        out.push(canon(bits, mt));
    }
    out
}

unsafe extern "C" {
    #[cfg(not(all(windows, target_arch = "x86_64")))]
    fn _setjmp(buf: *mut u8) -> i32;
    #[cfg(all(windows, target_arch = "x86_64"))]
    fn _setjmp(buf: *mut u8, frame: *mut u8) -> i32;
}

/// `setjmp(buf)`, then `body`. Returns 0 once `body` returns, or the value a
/// `longjmp` to `buf` delivered; `body`'s frames are then gone.
///
/// The runtime's throw longjmps to the buffer of the innermost handler, so
/// this frame has to stay live for as long as the try body runs, which is why
/// the body runs inside it.
///
/// # Safety
/// `buf` must be a jump buffer from `rayzor_exception_push_handler`.
#[inline(never)]
pub(super) unsafe fn setjmp_region(buf: *mut u8, body: &mut dyn FnMut()) -> i32 {
    #[cfg(not(all(windows, target_arch = "x86_64")))]
    let r = unsafe { _setjmp(buf) };
    #[cfg(all(windows, target_arch = "x86_64"))]
    let r = unsafe { _setjmp(buf, std::ptr::null_mut()) };
    if r == 0 {
        body();
        0
    } else {
        r
    }
}

/// A process symbol by name, for externs the runtime does not register.
pub(super) fn dlsym(name: &str) -> Option<usize> {
    #[cfg(unix)]
    {
        let c = std::ffi::CString::new(name).ok()?;
        let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c.as_ptr()) };
        (!p.is_null()).then_some(p as usize)
    }
    #[cfg(not(unix))]
    {
        let _ = name;
        None
    }
}
