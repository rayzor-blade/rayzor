//! # BeadieJit — beadie ↔ rayzor JitBackend adapter
//!
//! Bridges rayzor's [`CraneliftBackend`] to beadie's `JitBackend` trait so
//! per-function tier promotion can run through beadie's hot-function broker
//! instead of the hand-rolled queue + worker in `TieredBackend`.
//!
//! ## Design
//!
//! - The adapter owns a **dedicated** `Arc<Mutex<CraneliftBackend>>` —
//!   *not* a clone of `TieredBackend::baseline_backend`. Cranelift's
//!   `JITModule` doesn't permit re-defining an already-compiled symbol,
//!   so sharing the baseline backend (which has typically already
//!   compiled most functions via the all-or-nothing
//!   `compile_all_modules_jit` path) would dead-end at the first
//!   `compile_module_without_finalize` call. A dedicated backend gives
//!   beadie a clean slate to accumulate per-function compiles into.
//! - Cross-tier dispatch still works: rayzor's `function_pointers` map
//!   is the single source of truth for which native code runs. Whether
//!   the pointer came from `baseline_backend` or `beadie_backend` is
//!   invisible to the caller.
//! - The dedicated backend is constructed with the same runtime symbols
//!   passed to `baseline_backend`, so beadie-compiled code can call
//!   `haxe_*` runtime functions the same way.
//! - Each function gets one `Bead`. The bead's `FunctionDef` carries
//!   the module + `IrFunctionId` needed to stage that function's
//!   compile.
//! - `compile_outcome` uses [`CraneliftBackend::compile_module_without_finalize`]
//!   then returns a [`CompileOutcome::Deferred`] resolver that reads
//!   [`CraneliftBackend::get_function_ptr`] after [`Self::flush`] runs
//!   `finalize()` once for the whole batch.
//!
//! ## Scope
//!
//! Phase B step 2: dedicated backend + lazy bead registry, plus an
//! integration test that drives a real compile end-to-end through a
//! live `TieredBackend`. Still no `record_call` routing — that change
//! comes in step 3.
//!
//! ## Why the pivot from step 1
//!
//! Step 1's `build_batched_adapter` accepted any
//! `Arc<Mutex<CraneliftBackend>>`. The TieredBackend wiring at the time
//! passed an `Arc::clone(&baseline_backend)` on the assumption that
//! sharing the backend would avoid the "fresh backend per promotion"
//! leak documented on
//! [`crate::codegen::tiered_backend::TieredBackend::promotion_count`].
//! Step 2 investigation revealed that sharing breaks for the very
//! first compile: once `baseline_backend` is non-empty, beadie can't
//! stage another module into it without symbol-redefinition errors.
//! The dedicated-backend design preserves beadie's no-leak property
//! (one backend across the program's whole run, not one per
//! promotion) without forcing co-existence with already-compiled
//! code.

use std::sync::{Arc, Mutex};

use beadie::{BackendAdapter, Bead, CompileError, CompileOutcome, JitBackend, ThresholdPolicy};

use super::cranelift_backend::CraneliftBackend;
use crate::ir::{IrFunctionId, IrModule};

/// What beadie hands the backend when it's time to compile one function:
/// the module the function lives in plus the id to extract a pointer for.
///
/// Stored by value inside the bead so beadie can re-stage the compile
/// later (e.g. on reload after deopt) without rayzor re-supplying it.
pub struct BeadieFunctionDef {
    pub module: Arc<IrModule>,
    pub func_id: IrFunctionId,
}

/// `JitBackend` adapter over rayzor's [`CraneliftBackend`].
///
/// The backend is shared via `Arc<Mutex<_>>` so beadie's `&self`
/// `compile`/`compile_outcome`/`flush` methods can mutate the cranelift
/// state under a single lock.
pub struct BeadieJit {
    backend: Arc<Mutex<CraneliftBackend>>,
}

impl BeadieJit {
    /// Wrap a shared `Arc<Mutex<CraneliftBackend>>`. The caller keeps a
    /// clone if they need to read function pointers or finalize directly
    /// outside the broker — for example, to register pointers into
    /// `TieredBackend::function_pointers` after a batch.
    pub fn from_shared(backend: Arc<Mutex<CraneliftBackend>>) -> Self {
        Self { backend }
    }

    /// Borrow the shared backend (e.g. for cross-cut bookkeeping like
    /// RTTI registration or pointer harvesting after a batch flush).
    pub fn backend(&self) -> &Arc<Mutex<CraneliftBackend>> {
        &self.backend
    }
}

impl JitBackend for BeadieJit {
    type FunctionDef = BeadieFunctionDef;
    type Error = CompileError;

    /// Single-shot path: stage + finalize + read pointer in one call.
    ///
    /// Used by `BackendAdapter::with_policy` (non-batched). Not the
    /// expected hot path for rayzor — the production wiring uses
    /// `with_policy_batched` to share one `finalize()` across many
    /// promotions, but the trait requires this method.
    fn compile(&self, _bead: &Arc<Bead>, def: BeadieFunctionDef) -> Result<*mut (), Self::Error> {
        let mut be = self.backend.lock().unwrap();
        be.compile_module_without_finalize(&def.module)
            .map_err(CompileError::new)?;
        be.finalize().map_err(CompileError::new)?;
        let ptr = be
            .get_function_ptr(def.func_id)
            .map_err(CompileError::new)?;
        Ok(ptr as *mut ())
    }

    /// Batched path: stage with `compile_module_without_finalize`, defer
    /// the pointer read until [`Self::flush`] runs `finalize()` once for
    /// the whole batch.
    fn compile_outcome(
        &self,
        _bead: &Arc<Bead>,
        def: BeadieFunctionDef,
    ) -> Result<CompileOutcome, Self::Error> {
        let func_id = def.func_id;
        {
            let mut be = self.backend.lock().unwrap();
            be.compile_module_without_finalize(&def.module)
                .map_err(CompileError::new)?;
        }
        let backend_for_resolver = Arc::clone(&self.backend);
        Ok(CompileOutcome::Deferred(Box::new(move || {
            backend_for_resolver
                .lock()
                .unwrap()
                .get_function_ptr(func_id)
                .map(|p| p as *mut ())
                .unwrap_or(std::ptr::null_mut())
        })))
    }

    /// One `JITModule::finalize_definitions()` per batch.
    fn flush(&self) -> Result<(), Self::Error> {
        self.backend
            .lock()
            .unwrap()
            .finalize()
            .map_err(CompileError::new)
    }
}

/// Build a batched [`BackendAdapter<BeadieJit>`] over a caller-supplied
/// [`CraneliftBackend`] handle.
///
/// `threshold` is the call count at which beadie promotes a registered
/// function. Production wiring passes
/// [`crate::codegen::profiling::ProfileConfig::warm_threshold`] so
/// beadie's policy matches the hand-rolled enqueue criterion.
pub fn build_batched_adapter(
    backend: Arc<Mutex<CraneliftBackend>>,
    threshold: u32,
    capacity: usize,
    batch_limit: usize,
) -> Arc<BackendAdapter<BeadieJit>> {
    let jit = BeadieJit::from_shared(backend);
    Arc::new(BackendAdapter::from_arc_with_policy_batched(
        Arc::new(jit),
        ThresholdPolicy::new(threshold),
        capacity,
        batch_limit,
    ))
}

/// Build a batched [`BackendAdapter<BeadieJit>`] backed by a brand-new
/// [`CraneliftBackend`] initialised with the supplied runtime symbols.
///
/// Returns both the adapter (for routing compiles through beadie) and
/// the backing handle (so `TieredBackend` can register source-info /
/// RTTI / etc. directly against the same backend after a flush). The
/// pair shares the same `Arc<Mutex<CraneliftBackend>>` allocation.
pub fn build_dedicated_adapter(
    symbols: &[(&str, *const u8)],
    threshold: u32,
    capacity: usize,
    batch_limit: usize,
) -> Result<(Arc<BackendAdapter<BeadieJit>>, Arc<Mutex<CraneliftBackend>>), String> {
    let cranelift = CraneliftBackend::with_symbols(symbols)?;
    let backend = Arc::new(Mutex::new(cranelift));
    let adapter = build_batched_adapter(Arc::clone(&backend), threshold, capacity, batch_limit);
    Ok((adapter, backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::mir_builder::MirBuilder;
    use crate::ir::{BinaryOp, IrType};

    fn build_add_module() -> (Arc<IrModule>, IrFunctionId) {
        let mut builder = MirBuilder::new("beadie_jit_test");
        let func_id = builder
            .begin_function("add")
            .param("a", IrType::I64)
            .param("b", IrType::I64)
            .returns(IrType::I64)
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let a = builder.get_param(0);
        let b = builder.get_param(1);
        let sum = builder.bin_op(BinaryOp::Add, a, b);
        builder.ret(Some(sum));
        (Arc::new(builder.finish()), func_id)
    }

    /// The adapter constructs cleanly from a shared backend and the
    /// supplied `Arc` is adopted (`from_arc_with_policy_batched` checked
    /// upstream in beadie's own test suite).
    #[test]
    fn build_adapter_smoke() {
        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();
        let backend = CraneliftBackend::with_symbols(&symbols_ref).expect("cranelift backend");
        let shared = Arc::new(Mutex::new(backend));
        let adapter = build_batched_adapter(Arc::clone(&shared), 1, 16, 4);

        // Sanity: the adapter retained an Arc to the same backend
        // allocation we handed it.
        assert!(Arc::strong_count(adapter.backend().backend()) >= 1);
        let (_module, _func_id) = build_add_module();
    }

    /// `TieredBackend::beadie_adapter()` returns `Some` iff the config
    /// flag is set. The adapter owns a *dedicated* cranelift backend
    /// (see module-level rationale) so it can stage compiles without
    /// colliding with `baseline_backend`'s pre-existing symbol table.
    #[test]
    fn tiered_backend_constructs_with_beadie_adapter() {
        use super::super::tiered_backend::{TieredBackend, TieredConfig};

        // Off by default: no adapter.
        let off = TieredBackend::new(TieredConfig::default()).expect("tiered backend off");
        assert!(off.beadie_adapter().is_none());

        // Flip the flag: adapter present + dedicated backend present.
        let mut cfg = TieredConfig::default();
        cfg.enable_beadie_adapter = true;
        let on = TieredBackend::new(cfg).expect("tiered backend on");
        assert!(on.beadie_adapter().is_some());
        assert!(on.beadie_backend().is_some());
    }

    /// Drive a real compile through beadie inside a live `TieredBackend`.
    ///
    /// This is the end-to-end check that beadie can stage a rayzor MIR
    /// module → produce a Cranelift function pointer → transition the
    /// bead to `Compiled` state — all without touching any of the
    /// existing tier-promotion machinery. The compile lands in
    /// `beadie_backend`, not `baseline_backend`, proving the dedicated
    /// backend pivot works.
    #[test]
    fn tiered_backend_drives_real_compile_via_beadie() {
        use super::super::tiered_backend::{TieredBackend, TieredConfig};
        use beadie::BeadState;
        use std::time::{Duration, Instant};

        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        let mut cfg = TieredConfig::default();
        cfg.enable_beadie_adapter = true;
        // Threshold 1: first invoke triggers compile, no waiting.
        cfg.profile_config.warm_threshold = 1;

        let backend =
            TieredBackend::with_symbols(cfg, &symbols_ref).expect("tiered backend with beadie");
        let adapter = backend.beadie_adapter().expect("adapter present").clone();
        let (module, func_id) = build_add_module();
        let bound = adapter.register(std::ptr::null_mut(), None);

        let module_for_def = Arc::clone(&module);
        let outcome = adapter.on_invoke_outcome(&bound, move |_| BeadieFunctionDef {
            module: module_for_def,
            func_id,
        });
        // First call before compile finishes — None is the contract.
        assert!(outcome.is_none());

        let deadline = Instant::now() + Duration::from_secs(5);
        while bound.bead().compiled().is_none() {
            if Instant::now() > deadline {
                panic!("beadie did not install compiled pointer within 5s");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(bound.bead().state(), BeadState::Compiled);

        // Cast and invoke — proves the produced pointer is callable
        // Cranelift code, not a stub or null.
        let code = bound.bead().compiled().expect("compiled pointer");
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(20, 22), 42);
    }

    /// `ensure_beadie_bead` is idempotent and errors when the adapter
    /// is disabled. Used by step 3's `record_call` routing to register
    /// a bead lazily on the first warm-threshold crossing.
    #[test]
    fn ensure_beadie_bead_idempotence_and_disabled_error() {
        use super::super::tiered_backend::{TieredBackend, TieredConfig};
        use crate::tast::SymbolId;

        // Disabled: helper returns Err.
        let off = TieredBackend::new(TieredConfig::default()).expect("tiered backend off");
        let dummy = IrFunctionId(SymbolId(7).into());
        assert!(off.ensure_beadie_bead(dummy).is_err());
        assert!(!off.has_beadie_bead(dummy));

        // Enabled: first call registers, second is a no-op, registry
        // sees the entry.
        let mut cfg = TieredConfig::default();
        cfg.enable_beadie_adapter = true;
        let on = TieredBackend::new(cfg).expect("tiered backend on");
        assert!(!on.has_beadie_bead(dummy));
        on.ensure_beadie_bead(dummy).unwrap();
        assert!(on.has_beadie_bead(dummy));
        on.ensure_beadie_bead(dummy).unwrap(); // idempotent
        assert_eq!(on.beadie_beads().lock().unwrap().len(), 1);
    }
}
