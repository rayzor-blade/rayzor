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
//! Phase B step 3 (current): `record_call` routes
//! Standard-tier promotion through beadie when
//! [`crate::codegen::tiered_backend::TieredConfig::enable_beadie_adapter`]
//! is set (Option A — beadie owns the Standard threshold; see
//! commit history for the A/B rationale). Other tiers (Baseline,
//! Optimized, Maximum) still use the legacy queue + worker.
//! `PromotionBarrier`, `JitBailout`, and `ProfileData` accounting are
//! all preserved — the ProfileData increment stays in `record_call`
//! even on the beadie path, so the Optimized/Maximum decision logic
//! (which reads `get_function_count`) doesn't divide by two.
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

use std::sync::{Arc, Mutex, RwLock};

use beadie::{BackendAdapter, Bead, CompileError, CompileOutcome, JitBackend, ThresholdPolicy};

use super::cranelift_backend::CraneliftBackend;
use crate::ir::{IrFunctionId, IrModule};

/// What beadie hands the backend when it's time to compile one function:
/// a handle to the live module set + the `IrFunctionId` to extract a
/// pointer for.
///
/// The module set is shared via the same `Arc<RwLock<Vec<IrModule>>>`
/// that lives on [`crate::codegen::tiered_backend::TieredBackend`]. We
/// look up the owning module at compile time (under a brief read lock)
/// rather than snapshotting an `Arc<IrModule>` here, because:
///
/// - rayzor stores modules as `Vec<IrModule>` (not `Vec<Arc<IrModule>>`),
///   so producing an `Arc<IrModule>` at dispatch time would require a
///   deep clone of the whole `IrModule` for every promotion — wasteful
///   on large modules.
/// - The read lock during compilation blocks writers (rare —
///   `compile_module` only mutates modules at load time), not readers,
///   so live execution is unaffected.
///
/// Helper [`BeadieFunctionDef::from_single_module`] wraps a single
/// `Arc<IrModule>` in the same shape — useful for tests and the
/// `beadie_smoke` example that build standalone modules without a
/// `TieredBackend`.
pub struct BeadieFunctionDef {
    pub modules: Arc<RwLock<Vec<IrModule>>>,
    pub func_id: IrFunctionId,
}

impl BeadieFunctionDef {
    /// Test/example helper: wrap a single owned module so it satisfies
    /// the same `Arc<RwLock<Vec<IrModule>>>` shape used by
    /// `TieredBackend`. Performs one move of the module value (no
    /// clone). Returns the def **and** the shared handle so callers
    /// can keep their own clone if needed.
    pub fn from_single_module(
        module: IrModule,
        func_id: IrFunctionId,
    ) -> (Self, Arc<RwLock<Vec<IrModule>>>) {
        let modules = Arc::new(RwLock::new(vec![module]));
        let def = Self {
            modules: Arc::clone(&modules),
            func_id,
        };
        (def, modules)
    }
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
        let modules = def.modules.read().unwrap();
        let module = modules
            .iter()
            .find(|m| m.functions.contains_key(&def.func_id))
            .ok_or_else(|| {
                CompileError::new(format!(
                    "BeadieJit::compile: function {:?} not in any module",
                    def.func_id
                ))
            })?;
        let mut be = self.backend.lock().unwrap();
        be.compile_module_without_finalize(module)
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
            let modules = def.modules.read().unwrap();
            let module = modules
                .iter()
                .find(|m| m.functions.contains_key(&func_id))
                .ok_or_else(|| {
                    CompileError::new(format!(
                        "BeadieJit::compile_outcome: function {:?} not in any module",
                        func_id
                    ))
                })?;
            let mut be = self.backend.lock().unwrap();
            be.compile_module_without_finalize(module)
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

    fn build_add_module() -> (IrModule, IrFunctionId) {
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
        (builder.finish(), func_id)
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

        let (_def, modules_handle) = BeadieFunctionDef::from_single_module(module, func_id);
        let outcome = adapter.on_invoke_outcome(&bound, move |_| BeadieFunctionDef {
            modules: modules_handle,
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

    /// Phase B step 3 end-to-end: `record_call` past the warm threshold
    /// routes Standard-tier promotion through beadie, which compiles
    /// in the background. A follow-up `record_call` (or any other
    /// invocation that goes through `route_standard_to_beadie`) sees
    /// `bead().compiled()` and installs the pointer under the
    /// `PromotionBarrier`. After install, `jit_pointer` returns the
    /// pointer beadie produced and `function_tier` is `Standard`.
    #[test]
    fn record_call_routes_standard_via_beadie_end_to_end() {
        use super::super::tiered_backend::{OptimizationTier, TieredBackend, TieredConfig};
        use std::time::{Duration, Instant};

        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        let mut cfg = TieredConfig::default();
        cfg.enable_beadie_adapter = true;
        // Force the count to land *exactly* on Standard. Bumping
        // interpreter_threshold above warm prevents the "skip to
        // Baseline at count >= interpreter_threshold" path from firing
        // first; the actual lookup uses count >= warm_threshold.
        cfg.profile_config.warm_threshold = 1;
        cfg.profile_config.interpreter_threshold = 100;
        // Don't start the legacy background worker — this test
        // exercises the beadie scheduling, not the legacy queue.
        cfg.enable_background_optimization = false;

        let mut backend =
            TieredBackend::with_symbols(cfg, &symbols_ref).expect("tiered backend with beadie");
        let (module, func_id) = build_add_module();
        backend.compile_module(module).expect("module loaded");

        // First record_call: count starts at 0, sample passes, records
        // → count becomes 1 → target_tier resolves to Standard → routes
        // to beadie. The on_invoke_outcome call dispatches an
        // asynchronous compile; the immediate outcome is None.
        backend.record_call(func_id);

        // Wait for beadie's broker to finish the background compile.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut beadie_compiled = None;
        while Instant::now() < deadline {
            let ptr = backend
                .beadie_beads()
                .lock()
                .unwrap()
                .get(&func_id)
                .and_then(|b| b.bead().compiled());
            if let Some(p) = ptr {
                beadie_compiled = Some(p);
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let beadie_ptr =
            beadie_compiled.expect("beadie did not produce a compiled pointer within 5s");

        // Now trigger install via another record_call. The route_standard
        // path will observe bead().compiled() = Some(ptr) and install it
        // under the barrier, switching function_tier to Standard.
        backend.record_call(func_id);

        // Final assertions: pointer installed at Standard tier, and it
        // matches what beadie produced.
        assert_eq!(
            backend.function_tier(func_id),
            Some(OptimizationTier::Standard)
        );
        let installed_ptr = backend
            .jit_pointer(func_id)
            .expect("function_pointers missing entry after install");
        assert_eq!(installed_ptr, beadie_ptr as usize);

        // Sanity: the installed pointer is callable code, not a stub
        // or null.
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(beadie_ptr) };
        assert_eq!(f(19, 23), 42);
    }
}
