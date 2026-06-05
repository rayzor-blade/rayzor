//! `rayzor debug lldb` — launch a Haxe file under lldb with crash handlers pre-armed.
//!
//! Generates an lldb command script that:
//!   - intercepts SIGTRAP/SIGSEGV/SIGABRT and stops cleanly
//!   - on stop, dumps `bt 20` + the architecturally-relevant registers
//!   - keeps the lldb prompt open so you can poke around
//!
//! macOS only — we use `process handle <sig>` syntax + arm64 register
//! names directly. On other platforms the command prints a friendly
//! "not supported here" message and exits non-zero.

use super::DebugCommands;
use anyhow::{anyhow, Result};

#[cfg(not(target_os = "macos"))]
pub fn execute(_cmd: DebugCommands) -> Result<()> {
    eprintln!(
        "rayzor debug lldb: macOS-only command. On Linux, attach gdb manually:\n\
         \n\
         RAYZOR_DUMP_JIT_MAP=1 RAYZOR_DUMP_ALLOC_AT_EXIT=1 \\\n\
             gdb --args /path/to/rayzor run <file> -- <args>"
    );
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
pub fn execute(cmd: DebugCommands) -> Result<()> {
    let DebugCommands::Lldb {
        file,
        malloc_stack_log,
        program_args,
    } = cmd
    else {
        return Err(anyhow!("debug::lldb::execute called with non-Lldb variant"));
    };

    let self_exe = std::env::current_exe()?;
    let file_abs = file.canonicalize().unwrap_or(file.clone());

    // Build the lldb command file
    let mut script = String::new();
    script.push_str("process handle SIGTRAP -p true -s true -n true\n");
    script.push_str("process handle SIGSEGV -p true -s true -n true\n");
    script.push_str("process handle SIGBUS  -p true -s true -n true\n");
    script.push_str("process handle SIGABRT -p true -s true -n true\n");
    script.push_str("# auto-dump backtrace + registers when we stop on a crash\n");
    script.push_str(
        "target stop-hook add -o \"bt 20\" -o \"register read x0 x1 x2 x3 x19 x20 x21 x22 x23 x24 x25 fp sp pc lr\"\n",
    );

    // Launch line: process launch -- run <file> [-- <args>]
    let mut launch = String::from("process launch --");
    launch.push_str(" run ");
    launch.push_str(&shell_quote(&file_abs.to_string_lossy()));
    if !program_args.is_empty() {
        launch.push_str(" --");
        for a in &program_args {
            launch.push(' ');
            launch.push_str(&shell_quote(a));
        }
    }
    launch.push('\n');
    script.push_str(&launch);
    // After the program finishes (or stops), drop to the lldb prompt
    // rather than auto-quitting so the user can inspect state.
    // (no quit at end on purpose)

    // Write to a temp file
    let mut script_path = std::env::temp_dir();
    script_path.push(format!("rayzor-lldb-{}.txt", std::process::id()));
    std::fs::write(&script_path, &script)?;
    println!("[lldb] script: {}", script_path.display());

    // Spawn lldb
    let mut child = std::process::Command::new("lldb");
    child.arg("-s").arg(&script_path).arg(&self_exe);
    child.env("RAYZOR_DUMP_JIT_MAP", "1");
    child.env("RAYZOR_DUMP_ALLOC_AT_EXIT", "1");
    if malloc_stack_log {
        child.env("MallocStackLogging", "1");
        child.env("MallocStackLoggingNoCompact", "1");
    }
    let status = child.status()?;
    if let Some(code) = status.code() {
        std::process::exit(code);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn shell_quote(s: &str) -> String {
    // Conservative: wrap in double quotes if it contains anything funky.
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '-' | '_'))
    {
        s.to_string()
    } else {
        let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{escaped}\"")
    }
}
