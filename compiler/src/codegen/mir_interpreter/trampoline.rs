//! Code addresses for interpreted functions.
//!
//! Compiled code and the runtime call functions through code pointers: vtable
//! entries, closure objects, callbacks. An interpreted function gets a
//! 16-byte stub that puts its own address in a scratch register and jumps to
//! one assembly entry. The entry asks the dispatcher what the stub stands for:
//! the function's compiled code, which it tail-jumps to with the arguments
//! untouched, or nothing, in which case the dispatcher has already run the
//! function in the interpreter from the saved argument registers.
//!
//! Stubs are written once and never patched; installing compiled code only
//! updates the dispatcher's table.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::ir::IrFunctionId;

use super::Program;

const CHUNK: usize = 64 * 1024;
const HEADER: usize = 16;
const STUB: usize = 16;
const STUBS_PER_CHUNK: usize = (CHUNK - HEADER) / STUB;

/// What a stub stands for.
pub(super) struct Target {
    pub program: Arc<Program>,
    pub module: usize,
    pub func: IrFunctionId,
    /// Compiled code for the function, once there is some.
    pub forward: AtomicUsize,
}

struct Chunk {
    base: usize,
    used: usize,
    _map: memmap2::Mmap,
}

#[derive(Default)]
struct Registry {
    chunks: Vec<Chunk>,
    by_function: HashMap<(usize, IrFunctionId), usize>,
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(Default::default)
}

fn targets() -> &'static RwLock<HashMap<usize, Arc<Target>>> {
    static T: OnceLock<RwLock<HashMap<usize, Arc<Target>>>> = OnceLock::new();
    T.get_or_init(Default::default)
}

/// The function a stub address stands for.
pub(super) fn target_of(addr: usize) -> Option<Arc<Target>> {
    targets().read().unwrap().get(&addr).cloned()
}

/// The stub for `func` of the module at `module`, made on first request.
pub(super) fn address(
    program: &Arc<Program>,
    module: usize,
    func: IrFunctionId,
) -> Result<usize, String> {
    let mut reg = registry().lock().unwrap();
    if let Some(&addr) = reg.by_function.get(&(module, func)) {
        return Ok(addr);
    }
    if reg.chunks.last().is_none_or(|c| c.used == STUBS_PER_CHUNK) {
        let chunk = new_chunk()?;
        reg.chunks.push(chunk);
    }
    let chunk = reg.chunks.last_mut().unwrap();
    let addr = chunk.base + HEADER + chunk.used * STUB;
    chunk.used += 1;
    let forward = program.native(func).unwrap_or(0);
    targets().write().unwrap().insert(
        addr,
        Arc::new(Target {
            program: program.clone(),
            module,
            func,
            forward: AtomicUsize::new(forward),
        }),
    );
    reg.by_function.insert((module, func), addr);
    Ok(addr)
}

/// Send every stub for `func` straight to its compiled code.
pub(super) fn forward(func: IrFunctionId, code: usize) {
    for t in targets().read().unwrap().values() {
        if t.func == func {
            t.forward.store(code, Ordering::Release);
        }
    }
}

fn new_chunk() -> Result<Chunk, String> {
    let entry = entry_address().ok_or("no trampoline entry on this target")?;
    let mut map = memmap2::MmapMut::map_anon(CHUNK)
        .map_err(|e| format!("cannot map trampoline memory: {}", e))?;
    let base = map.as_ptr() as usize;
    map[..8].copy_from_slice(&(entry as u64).to_le_bytes());
    for i in 0..STUBS_PER_CHUNK {
        let at = HEADER + i * STUB;
        write_stub(&mut map[at..at + STUB], base + at, base);
    }
    let map = map
        .make_exec()
        .map_err(|e| format!("cannot make trampoline memory executable: {}", e))?;
    flush_icache(base, CHUNK);
    Ok(Chunk {
        base,
        used: 0,
        _map: map,
    })
}

/// `adr x16, .; ldr x17, <header>; br x17`.
#[cfg(target_arch = "aarch64")]
fn write_stub(out: &mut [u8], at: usize, header: usize) {
    let rel = (header as i64 - (at as i64 + 4)) >> 2;
    let words: [u32; 4] = [
        0x1000_0010,
        0x5800_0000 | (((rel as u32) & 0x7_ffff) << 5) | 17,
        0xd61f_0220,
        0xd420_0000,
    ];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
}

/// `lea r11, [rip - 7]; jmp [rip + <header>]`.
#[cfg(target_arch = "x86_64")]
fn write_stub(out: &mut [u8], at: usize, header: usize) {
    let disp = (header as i64 - (at as i64 + 13)) as i32;
    out[..7].copy_from_slice(&[0x4c, 0x8d, 0x1d, 0xf9, 0xff, 0xff, 0xff]);
    out[7..9].copy_from_slice(&[0xff, 0x25]);
    out[9..13].copy_from_slice(&disp.to_le_bytes());
    out[13..16].fill(0xcc);
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn write_stub(_out: &mut [u8], _at: usize, _header: usize) {}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
fn flush_icache(start: usize, len: usize) {
    unsafe extern "C" {
        fn sys_icache_invalidate(start: *const std::ffi::c_void, size: usize);
    }
    unsafe { sys_icache_invalidate(start as *const _, len) };
}

#[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
fn flush_icache(start: usize, len: usize) {
    use std::arch::asm;
    let ctr: u64;
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr) };
    let dline = 4usize << ((ctr >> 16) & 0xf);
    let iline = 4usize << (ctr & 0xf);
    let end = start + len;
    let mut a = start & !(dline - 1);
    while a < end {
        unsafe { asm!("dc cvau, {}", in(reg) a) };
        a += dline;
    }
    unsafe { asm!("dsb ish") };
    let mut a = start & !(iline - 1);
    while a < end {
        unsafe { asm!("ic ivau, {}", in(reg) a) };
        a += iline;
    }
    unsafe { asm!("dsb ish", "isb") };
}

#[cfg(not(target_arch = "aarch64"))]
fn flush_icache(_start: usize, _len: usize) {}

/// The argument registers a native caller passed, as the entry saves them.
#[repr(C)]
pub(super) struct Saved {
    /// x0-x7 and x8 / rdi, rsi, rdx, rcx, r8, r9 / rcx, rdx, r8, r9.
    pub ints: [u64; 9],
    /// d0-d7 / xmm0-xmm7 / xmm0-xmm3.
    pub floats: [u64; 8],
    /// The caller's first stack argument.
    pub stack: *const u8,
    _pad: u64,
}

/// Run the function a stub stands for, or say where its compiled code is.
extern "C" fn dispatch(stub: usize, saved: *mut Saved) -> usize {
    let Some(target) = target_of(stub) else {
        eprintln!(
            "rayzor: call through an unknown interpreter trampoline {:#x}",
            stub
        );
        std::process::abort();
    };
    let code = target.forward.load(Ordering::Acquire);
    if code != 0 {
        return code;
    }
    let saved = unsafe { &mut *saved };
    if let Err(e) = super::enter_from_native(&target, saved) {
        eprintln!("rayzor: interpreter error in a native callback: {}", e);
        std::process::abort();
    }
    0
}

#[cfg(target_vendor = "apple")]
macro_rules! entry_symbol {
    () => {
        "_rayzor_interp_trampoline_entry"
    };
}
#[cfg(not(target_vendor = "apple"))]
macro_rules! entry_symbol {
    () => {
        "rayzor_interp_trampoline_entry"
    };
}

#[cfg(target_arch = "aarch64")]
std::arch::global_asm!(
    ".text",
    ".p2align 2",
    concat!(".globl ", entry_symbol!()),
    concat!(entry_symbol!(), ":"),
    "stp x29, x30, [sp, #-16]!",
    "mov x29, sp",
    "sub sp, sp, #160",
    "stp x0, x1, [sp, #0]",
    "stp x2, x3, [sp, #16]",
    "stp x4, x5, [sp, #32]",
    "stp x6, x7, [sp, #48]",
    "str x8, [sp, #64]",
    "stp d0, d1, [sp, #72]",
    "stp d2, d3, [sp, #88]",
    "stp d4, d5, [sp, #104]",
    "stp d6, d7, [sp, #120]",
    "add x9, x29, #16",
    "str x9, [sp, #136]",
    "mov x0, x16",
    "mov x1, sp",
    "bl {dispatch}",
    "cbnz x0, 2f",
    "ldr x0, [sp, #0]",
    "ldr d0, [sp, #72]",
    "mov sp, x29",
    "ldp x29, x30, [sp], #16",
    "ret",
    "2:",
    "mov x17, x0",
    "ldp x0, x1, [sp, #0]",
    "ldp x2, x3, [sp, #16]",
    "ldp x4, x5, [sp, #32]",
    "ldp x6, x7, [sp, #48]",
    "ldr x8, [sp, #64]",
    "ldp d0, d1, [sp, #72]",
    "ldp d2, d3, [sp, #88]",
    "ldp d4, d5, [sp, #104]",
    "ldp d6, d7, [sp, #120]",
    "mov sp, x29",
    "ldp x29, x30, [sp], #16",
    "br x17",
    dispatch = sym dispatch,
);

#[cfg(all(target_arch = "x86_64", not(windows)))]
std::arch::global_asm!(
    ".text",
    ".p2align 4",
    concat!(".globl ", entry_symbol!()),
    concat!(entry_symbol!(), ":"),
    "push rbp",
    "mov rbp, rsp",
    "sub rsp, 160",
    "mov qword ptr [rsp], rdi",
    "mov qword ptr [rsp + 8], rsi",
    "mov qword ptr [rsp + 16], rdx",
    "mov qword ptr [rsp + 24], rcx",
    "mov qword ptr [rsp + 32], r8",
    "mov qword ptr [rsp + 40], r9",
    "movsd qword ptr [rsp + 72], xmm0",
    "movsd qword ptr [rsp + 80], xmm1",
    "movsd qword ptr [rsp + 88], xmm2",
    "movsd qword ptr [rsp + 96], xmm3",
    "movsd qword ptr [rsp + 104], xmm4",
    "movsd qword ptr [rsp + 112], xmm5",
    "movsd qword ptr [rsp + 120], xmm6",
    "movsd qword ptr [rsp + 128], xmm7",
    "lea rax, [rbp + 16]",
    "mov qword ptr [rsp + 136], rax",
    "mov rdi, r11",
    "mov rsi, rsp",
    "call {dispatch}",
    "test rax, rax",
    "jnz 2f",
    "mov rax, qword ptr [rsp]",
    "movsd xmm0, qword ptr [rsp + 72]",
    "mov rsp, rbp",
    "pop rbp",
    "ret",
    "2:",
    "mov r11, rax",
    "mov rdi, qword ptr [rsp]",
    "mov rsi, qword ptr [rsp + 8]",
    "mov rdx, qword ptr [rsp + 16]",
    "mov rcx, qword ptr [rsp + 24]",
    "mov r8, qword ptr [rsp + 32]",
    "mov r9, qword ptr [rsp + 40]",
    "movsd xmm0, qword ptr [rsp + 72]",
    "movsd xmm1, qword ptr [rsp + 80]",
    "movsd xmm2, qword ptr [rsp + 88]",
    "movsd xmm3, qword ptr [rsp + 96]",
    "movsd xmm4, qword ptr [rsp + 104]",
    "movsd xmm5, qword ptr [rsp + 112]",
    "movsd xmm6, qword ptr [rsp + 120]",
    "movsd xmm7, qword ptr [rsp + 128]",
    "mov rsp, rbp",
    "pop rbp",
    "jmp r11",
    dispatch = sym dispatch,
);

/// Win64 passes four arguments by position and the fifth at `[rsp + 40]` on
/// entry, past the caller's shadow space; the dispatcher call needs its own.
#[cfg(all(target_arch = "x86_64", windows))]
std::arch::global_asm!(
    ".text",
    ".p2align 4",
    concat!(".globl ", entry_symbol!()),
    concat!(entry_symbol!(), ":"),
    "push rbp",
    "mov rbp, rsp",
    "sub rsp, 192",
    "mov qword ptr [rsp + 32], rcx",
    "mov qword ptr [rsp + 40], rdx",
    "mov qword ptr [rsp + 48], r8",
    "mov qword ptr [rsp + 56], r9",
    "movsd qword ptr [rsp + 104], xmm0",
    "movsd qword ptr [rsp + 112], xmm1",
    "movsd qword ptr [rsp + 120], xmm2",
    "movsd qword ptr [rsp + 128], xmm3",
    "lea rax, [rbp + 48]",
    "mov qword ptr [rsp + 168], rax",
    "mov rcx, r11",
    "lea rdx, [rsp + 32]",
    "call {dispatch}",
    "test rax, rax",
    "jnz 2f",
    "mov rax, qword ptr [rsp + 32]",
    "movsd xmm0, qword ptr [rsp + 104]",
    "mov rsp, rbp",
    "pop rbp",
    "ret",
    "2:",
    "mov r11, rax",
    "mov rcx, qword ptr [rsp + 32]",
    "mov rdx, qword ptr [rsp + 40]",
    "mov r8, qword ptr [rsp + 48]",
    "mov r9, qword ptr [rsp + 56]",
    "movsd xmm0, qword ptr [rsp + 104]",
    "movsd xmm1, qword ptr [rsp + 112]",
    "movsd xmm2, qword ptr [rsp + 120]",
    "movsd xmm3, qword ptr [rsp + 128]",
    "mov rsp, rbp",
    "pop rbp",
    "jmp r11",
    dispatch = sym dispatch,
);

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn entry_address() -> Option<usize> {
    unsafe extern "C" {
        fn rayzor_interp_trampoline_entry();
    }
    Some(rayzor_interp_trampoline_entry as *const () as usize)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn entry_address() -> Option<usize> {
    None
}
