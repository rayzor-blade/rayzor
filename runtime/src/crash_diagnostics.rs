//! Fatal native crash diagnostics. The handler reads immutable records and
//! writes bytes without using the allocator, locks, or an unwinder.
use std::sync::atomic::{AtomicPtr, Ordering};

struct CodeRange {
    start: usize,
    end: usize,
    label: Box<[u8]>,
    next: *mut CodeRange,
}
static RANGES: AtomicPtr<CodeRange> = AtomicPtr::new(std::ptr::null_mut());

/// Publish before execution. Records stay alive for concurrent signal readers;
/// the newest registration wins when a JIT address is reused.
pub fn register_code(start: usize, size: usize, name: &str) {
    if start == 0 || size == 0 {
        return;
    }
    let node = Box::into_raw(Box::new(CodeRange {
        start,
        end: start.saturating_add(size),
        label: format!(" in {name} [Cranelift]")
            .into_bytes()
            .into_boxed_slice(),
        next: std::ptr::null_mut(),
    }));
    let mut head = RANGES.load(Ordering::Acquire);
    loop {
        unsafe {
            (*node).next = head;
        }
        match RANGES.compare_exchange_weak(head, node, Ordering::Release, Ordering::Acquire) {
            Ok(_) => break,
            Err(current) => head = current,
        }
    }
}

#[cfg(unix)]
unsafe fn write(bytes: &[u8]) {
    libc::write(libc::STDERR_FILENO, bytes.as_ptr().cast(), bytes.len());
}
#[cfg(unix)]
unsafe fn hex(value: usize) {
    let mut bytes = [b'0'; 2 + 2 * std::mem::size_of::<usize>()];
    bytes[1] = b'x';
    for i in 2..bytes.len() {
        let shift = (bytes.len() - 1 - i) * 4;
        bytes[i] = b"0123456789abcdef"[(value >> shift) & 15];
    }
    write(&bytes);
}
#[cfg(unix)]
unsafe fn location(pc: usize) {
    hex(pc);
    let mut node = RANGES.load(Ordering::Acquire);
    while !node.is_null() {
        let entry = &*node;
        if pc >= entry.start && pc < entry.end {
            write(&entry.label);
            write(b" +");
            hex(pc - entry.start);
            return;
        }
        node = entry.next;
    }
    write(b" [outside registered JIT code]");
}
#[cfg(unix)]
unsafe extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    write(b"rayzor: native crash signal=");
    write(match sig {
        libc::SIGSEGV => b"SIGSEGV",
        libc::SIGBUS => b"SIGBUS",
        libc::SIGILL => b"SIGILL",
        libc::SIGABRT => b"SIGABRT",
        libc::SIGTRAP => b"SIGTRAP",
        _ => b"unknown",
    });
    if !info.is_null() {
        write(b" fault=");
        hex((*info).si_addr() as usize);
    }
    let (pc, caller) = registers(ctx);
    write(b" pc=");
    location(pc);
    if caller != 0 {
        write(b" caller=");
        location(caller.saturating_sub(1));
    }
    write(b"\n");
    libc::_exit(128 + sig);
}
#[cfg(unix)]
unsafe fn registers(ctx: *mut libc::c_void) -> (usize, usize) {
    if ctx.is_null() {
        return (0, 0);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let mc = (*(ctx as *const libc::ucontext_t)).uc_mcontext;
        if !mc.is_null() {
            return ((*mc).__ss.__pc as usize, (*mc).__ss.__lr as usize);
        }
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        let mc = (*(ctx as *const libc::ucontext_t)).uc_mcontext;
        if !mc.is_null() {
            return ((*mc).__ss.__rip as usize, 0);
        }
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return (
            (*(ctx as *const libc::ucontext_t)).uc_mcontext.gregs[libc::REG_RIP as usize] as usize,
            0,
        );
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let mc = &(*(ctx as *const libc::ucontext_t)).uc_mcontext;
        return (mc.pc as usize, mc.regs[30] as usize);
    }
    #[allow(unreachable_code)]
    (0, 0)
}

/// The CLI opts in; embedders retain their own signal handlers.
pub fn install() {
    #[cfg(unix)]
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handler as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO | libc::SA_RESETHAND;
        libc::sigemptyset(&mut action.sa_mask);
        for sig in [
            libc::SIGSEGV,
            libc::SIGBUS,
            libc::SIGILL,
            libc::SIGABRT,
            libc::SIGTRAP,
        ] {
            libc::sigaction(sig, &action, std::ptr::null_mut());
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    #[test]
    fn fatal_signal_reports_address_and_keeps_signal_status() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "crash_diagnostics::tests::crash_child",
                "--ignored",
                "--nocapture",
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(128 + libc::SIGSEGV));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("rayzor: native crash signal="), "{stderr}");
        assert!(
            stderr.contains("fault=0x") && stderr.contains("pc=0x"),
            "{stderr}"
        );
        assert!(
            stderr.contains("Fixture.main [Cranelift] +0x0000000000000004"),
            "{stderr}"
        );
    }

    #[test]
    #[ignore = "subprocess used by fatal_signal_reports_address_and_keeps_signal_status"]
    fn crash_child() {
        super::install();
        super::register_code(0x1000, 8, "Fixture.main");
        unsafe {
            super::location(0x1004);
            libc::raise(libc::SIGSEGV);
        }
        panic!("fatal signal returned");
    }
}
