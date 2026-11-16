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
    /// `RAYZOR_ENUM_LOOKUP_DEBUG`: trace enum helper arguments and their type identities.
    enum_lookup_debug,
    "RAYZOR_ENUM_LOOKUP_DEBUG"
);
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
    /// `RAYZOR_LOWER_TRACE`: name each expression MIR lowering produced no
    /// value for -- a variable, unary, conditional or call -- and the probe
    /// that claimed each call.
    lower_trace,
    "RAYZOR_LOWER_TRACE"
);
cached_flag!(
    /// `RAYZOR_VECCALL_DIAG`: report direct calls taking vector arguments.
    veccall_diag,
    "RAYZOR_VECCALL_DIAG"
);
cached_flag!(
    /// `RAYZOR_NO_XMODULE_CTOR`: restore the unconditional one-argument value
    /// wrap in `lower_new`, disabling cross-module constructor recovery.
    /// Attributes a regression in one run instead of a rebuild and bisect.
    no_xmodule_ctor,
    "RAYZOR_NO_XMODULE_CTOR"
);
cached_flag!(
    /// `RAYZOR_NULL_INT_BOXED`: keep `Null<Int>` as a box instead of `NullInt`.
    /// Safe to run with; compile with `--no-cache`, the cache does not key on it.
    null_int_boxed,
    "RAYZOR_NULL_INT_BOXED"
);
cached_flag!(
    /// `RAYZOR_ERASED_CENSUS`: report scalars crossing erased generic slots
    /// without a word conversion. Report only; safe to run with.
    erased_census,
    "RAYZOR_ERASED_CENSUS"
);
/// Whether an erased generic slot holds a self-describing word rather than
/// raw bits. On by default; `RAYZOR_ERASED_WORDS=0` keeps raw slots, safe to
/// run with. Off for a WASM target, whose slots are 32-bit. Cached artifacts
/// are keyed by it.
pub fn erased_words() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let requested = *ON
        .get_or_init(|| std::env::var_os("RAYZOR_ERASED_WORDS").is_none_or(|value| value != "0"));
    requested && !RAW_SLOT_TARGET.load(std::sync::atomic::Ordering::Relaxed)
}

static RAW_SLOT_TARGET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Compile for a target that keeps raw erased slots. Set before lowering.
pub fn use_raw_slots() {
    RAW_SLOT_TARGET.store(true, std::sync::atomic::Ordering::Relaxed);
}
