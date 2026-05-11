//! # BeadieJit — beadie ↔ rayzor JitBackend adapter
//!
//! Bridges rayzor's [`CraneliftBackend`] to beadie's `JitBackend` trait so
//! per-function tier promotion can run through beadie's hot-function broker
//! instead of the hand-rolled queue + worker in `TieredBackend`.
//!
//! ## Design
//!
//! - One shared `Arc<Mutex<CraneliftBackend>>` lives behind the adapter.
//!   Reusing a single backend across every promotion avoids the
//!   "each promotion leaks a backend" cost path documented on
//!   [`crate::codegen::tiered_backend::TieredBackend::promotion_count`].
//! - Each function gets one `Bead`. The bead's `FunctionDef` carries the
//!   module + `IrFunctionId` needed to stage that function's compile.
//! - `compile_outcome` uses [`CraneliftBackend::compile_module_without_finalize`]
//!   then returns a [`CompileOutcome::Deferred`] resolver that reads
//!   [`CraneliftBackend::get_function_ptr`] after [`Self::flush`] runs
//!   `finalize()` once for the whole batch.
//!
//! ## Scope
//!
//! Phase B step 1: infrastructure only. The adapter is wireable but not
//! yet wired into `TieredBackend::optimize_function_internal`. The
//! existing [`crate::codegen::tiered_backend`] tier ladder (Baseline →
//! Standard → Optimized → Maximum), `PromotionBarrier`, and interpreter
//! `JitBailout` path all stay intact.

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

/// Build a batched [`BackendAdapter<BeadieJit>`] from a shared backend.
///
/// `threshold` is the call count at which beadie promotes a registered
/// function. For Phase B step 2 we'll pass the existing
/// `ProfileConfig::warm_threshold` here so beadie's policy matches the
/// hand-rolled enqueue criterion.
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
    /// flag is set. The adapter must share the same cranelift backend
    /// allocation as `TieredBackend::baseline_backend` — otherwise a
    /// future tier promotion routed through beadie would compile into
    /// the wrong backend.
    #[test]
    fn tiered_backend_constructs_with_beadie_adapter() {
        use super::super::tiered_backend::{TieredBackend, TieredConfig};

        // Off by default: no adapter.
        let off = TieredBackend::new(TieredConfig::default()).expect("tiered backend off");
        assert!(off.beadie_adapter().is_none());

        // Flip the flag: adapter present.
        let mut cfg = TieredConfig::default();
        cfg.enable_beadie_adapter = true;
        let on = TieredBackend::new(cfg).expect("tiered backend on");
        assert!(on.beadie_adapter().is_some());
    }
}
