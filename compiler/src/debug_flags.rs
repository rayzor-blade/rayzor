//! Debug flags read once.
//!
//! These gate `eprintln!` probes that sit inside per-node lowering helpers --
//! `lower_variable_expr` runs for every variable reference in the program,
//! `lower_new` for every allocation, `build_call_direct` for every call. A
//! `std::env::var` there is a scan of the environment block per AST node, paid
//! by every compile whether or not the flag is set.

macro_rules! cached_flag {
    ($(#[$m:meta])* $name:ident, $var:literal) => {
        $(#[$m])*
        pub fn $name() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var_os($var).is_some())
        }
    };
}

cached_flag!(
    /// `RAYZOR_WILDCARD_LOG`: trace wildcard static-import resolution.
    wildcard_log,
    "RAYZOR_WILDCARD_LOG"
);
cached_flag!(
    /// `RAYZOR_GLOBALS_DEBUG`: trace global/variable resolution.
    globals_debug,
    "RAYZOR_GLOBALS_DEBUG"
);
cached_flag!(
    /// `RAYZOR_CTOR_DEBUG`: trace constructor lowering.
    ctor_debug,
    "RAYZOR_CTOR_DEBUG"
);
cached_flag!(
    /// `RAYZOR_ALLOC_DEBUG`: trace allocation sizes.
    alloc_debug,
    "RAYZOR_ALLOC_DEBUG"
);
cached_flag!(
    /// `RAYZOR_PROBE_CALLTARGET`: name the callee chosen at each call.
    probe_calltarget,
    "RAYZOR_PROBE_CALLTARGET"
);
cached_flag!(
    /// `RAYZOR_VECCALL_DIAG`: report direct calls taking vector arguments.
    veccall_diag,
    "RAYZOR_VECCALL_DIAG"
);
