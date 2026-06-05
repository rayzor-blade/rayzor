//! `rayzor debug run` — invoke `rayzor run` with forensic env vars wired up.
//!
//! Sets the appropriate env vars from `runtime/src/profile.rs` etc. and
//! re-execs the current binary with the standard `run` subcommand. We use
//! exec/spawn rather than calling the run path in-process so the env vars
//! affect the same process the runtime initialises against (some of them
//! are read once at startup via `Once::call_once`).

use super::DebugCommands;
use anyhow::{anyhow, Result};
use std::process::Command;

pub fn execute(cmd: DebugCommands) -> Result<()> {
    let DebugCommands::Run {
        file,
        no_jit_map,
        no_backtrace,
        alloc_stats,
        alloc_graph,
        cpu_profile,
        heap_check,
        malloc_stack_log,
        preset,
        program_args,
    } = cmd
    else {
        return Err(anyhow!("debug::run::execute called with non-Run variant"));
    };

    let self_exe = std::env::current_exe()?;
    let mut child = Command::new(&self_exe);
    child.arg("run").arg(&file).arg("--preset").arg(&preset);

    // Warn about feature-gated flags when the binary wasn't built with
    // `--features profile`. We detect this at build time below.
    let profile_built = profile_feature_present();
    let wants_profile = alloc_stats || alloc_graph.is_some() || cpu_profile.is_some();
    if wants_profile && !profile_built {
        eprintln!(
            "[rayzor debug] warning: --alloc-stats/--alloc-graph/--cpu-profile require\n\
             a binary built with `--features profile`. Continuing without them.\n\
             Rebuild: cargo build --release --features profile -p rayzor"
        );
    }

    // JIT map dump (compiler/src/codegen/cranelift_backend.rs) +
    // file table dump (compiler/src/compilation.rs). Both gated by
    // RAYZOR_DUMP_JIT_MAP=1 with the per-PID file written automatically.
    if !no_jit_map {
        child.env("RAYZOR_DUMP_JIT_MAP", "1");
    }

    // sigaction crash handler with ucontext PC capture (macOS arm64) +
    // dladdr resolution. Gated by RAYZOR_DUMP_ALLOC_AT_EXIT=1 in the
    // current code (same atexit hook installs the sig handler).
    if !no_backtrace {
        child.env("RAYZOR_DUMP_ALLOC_AT_EXIT", "1");
    }

    if alloc_stats && profile_built {
        child.env("RAYZOR_DUMP_ALLOC_AT_EXIT", "1");
    }

    if let Some(rate) = alloc_graph {
        if profile_built {
            child.env("RAYZOR_ALLOC_GRAPH", "1");
            child.env("RAYZOR_ALLOC_GRAPH_RATE", rate.to_string());
            child.env("RAYZOR_DUMP_ALLOC_AT_EXIT", "1");
        }
    }

    if let Some(us) = cpu_profile {
        if profile_built {
            child.env("RAYZOR_CPU_PROFILE", "1");
            child.env("RAYZOR_CPU_PROFILE_US", us.to_string());
            child.env("RAYZOR_DUMP_ALLOC_AT_EXIT", "1");
        }
    }

    if heap_check {
        #[cfg(target_os = "macos")]
        child.env("RAYZOR_HEAP_CHECK", "1");
        #[cfg(not(target_os = "macos"))]
        eprintln!("[rayzor debug] --heap-check is macOS-only; ignored on this platform");
    }

    if malloc_stack_log {
        #[cfg(target_os = "macos")]
        {
            child.env("MallocStackLogging", "1");
            child.env("MallocStackLoggingNoCompact", "1");
        }
        #[cfg(not(target_os = "macos"))]
        eprintln!("[rayzor debug] --malloc-stack-log is macOS-only; ignored on this platform");
    }

    if !program_args.is_empty() {
        child.arg("--");
        for a in program_args {
            child.arg(a);
        }
    }

    // Forward stdio; bubble the exit code up
    let status = child.status()?;
    if let Some(code) = status.code() {
        std::process::exit(code);
    }
    Err(anyhow!("`rayzor run` exited via signal: {status:?}"))
}

/// Detect whether this binary was built with `--features profile`. The
/// detection mirrors what `peak_bench.sh` uses: look for the
/// `rayzor_dump_tensor_alloc_stats` symbol in our own binary. The symbol
/// is exported by `runtime/src/tensor.rs::rayzor_dump_tensor_alloc_stats`
/// and only emitted under the profile feature.
fn profile_feature_present() -> bool {
    // Cheap heuristic: try dlsym for the symbol on our own image. macOS
    // exposes symbols via `dlopen(NULL)` + `dlsym`.
    #[cfg(unix)]
    {
        unsafe {
            extern "C" {
                fn dlsym(handle: *mut std::ffi::c_void, symbol: *const i8)
                    -> *mut std::ffi::c_void;
            }
            let sym = std::ffi::CString::new("rayzor_dump_tensor_alloc_stats").unwrap();
            let rtld_default: *mut std::ffi::c_void = std::ptr::null_mut();
            let ptr = dlsym(rtld_default, sym.as_ptr());
            !ptr.is_null()
        }
    }
    #[cfg(not(unix))]
    {
        false
    }
}
