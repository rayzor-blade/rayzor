//! # BeadieJit — beadie ↔ rayzor JitBackend adapter
//!
//! Bridges rayzor's [`CraneliftBackend`] to beadie's `JitBackend` trait
//! so per-function tier promotion can run through beadie's hot-function
//! broker instead of the hand-rolled queue + worker in `TieredBackend`.
//!
//! ## Design (Phase B step 4: fresh-backend-per-compile)
//!
//! Each `compile_outcome` call spins up a **fresh**
//! [`CraneliftBackend`], compiles **all** loaded modules into it,
//! finalizes once, extracts the target function pointer, then
//! [`Box::leak`]s the backend so its JIT memory outlives the closure.
//! This matches the legacy
//! [`super::tiered_backend::TieredBackend::compile_all_at_tier`]
//! pattern documented at the `each leaks a backend` comment on
//! `TieredBackend::promotion_count`.
//!
//! ### Why this pattern, after the dedicated-backend pivot in step 2
//!
//! The dedicated-backend design from step 2 hit a hard wall on
//! cross-module calls: when module M1's function A calls function B in
//! module M2, beadie's compile_module_without_finalize(M1) declares B
//! as an `Import`, which fails at finalize unless cranelift can
//! resolve B by symbol name. With Box::leak'd full-module compiles,
//! all functions live in the same `JITModule` so cranelift resolves
//! them as internal symbols — exactly the mechanism that lets the
//! legacy `compile_all_at_tier` produce working code today.
//!
//! ### What rayzor gains from beadie under this pattern
//!
//! The compile mechanics are equivalent to legacy; **what beadie
//! brings is the broker** — threshold detection, batched scheduling,
//! and the broker-thread compile loop. `record_call` no longer talks
//! to `ProfileData` + `optimization_queue` for Standard-tier
//! promotion; it talks to `BackendAdapter::on_invoke_outcome`
//! instead. Same compile cost, different scheduling layer.
//!
//! ## Scope
//!
//! Step 3 wired `record_call`. Step 4 (this commit) fixes the
//! cross-module gap so the integration works for real Haxe programs
//! with multi-module call graphs. The
//! [`CraneliftBackend::with_symbols_lookup_and_opt`] constructor
//! added alongside this commit isn't strictly required by step 4 —
//! it's groundwork for a future "incremental shared-backend"
//! optimisation that would avoid leaking a fresh backend per
//! promotion.

use std::sync::{Arc, Mutex, RwLock};

use beadie::{BackendAdapter, Bead, CompileError, CompileOutcome, JitBackend, ThresholdPolicy};

use super::cranelift_backend::CraneliftBackend;
use crate::ir::{IrFunctionId, IrModule};

/// The compile unit beadie hands the backend at dispatch time: the
/// live module set + the target `IrFunctionId`.
///
/// `modules` is shared via the same `Arc<RwLock<Vec<IrModule>>>` that
/// lives on [`crate::codegen::tiered_backend::TieredBackend`]. Late
/// lookup under a brief read lock avoids deep-cloning the whole
/// `IrModule` per dispatch.
///
/// Helper [`BeadieFunctionDef::from_single_module`] wraps a standalone
/// module in the same shape for tests / smoke usage that don't have a
/// `TieredBackend` around.
pub struct BeadieFunctionDef {
    pub modules: Arc<RwLock<Vec<IrModule>>>,
    pub func_id: IrFunctionId,
}

impl BeadieFunctionDef {
    /// Test/example helper: wrap a single owned module so it satisfies
    /// the `Arc<RwLock<Vec<IrModule>>>` shape. Returns the def and a
    /// clone of the shared handle.
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

/// Snapshot of runtime symbols beadie reuses for every fresh backend
/// it spins up. Stored as `(String, usize)` so the slice can be
/// rebuilt as `&[(&str, *const u8)]` at each compile.
pub type RuntimeSymbolTable = Arc<Vec<(String, usize)>>;

/// `JitBackend` adapter that drives a fresh [`CraneliftBackend`] per
/// compile and leaks it. See module-level docs for the rationale.
///
/// Holds:
/// - `runtime_symbols` — the rayzor runtime symbol table to register
///   on every freshly-constructed cranelift backend.
/// - `opt_level` — the cranelift opt level to compile at. Set from
///   `OptimizationTier::Standard.cranelift_opt_level()` (= "speed")
///   when this adapter handles Standard-tier promotions.
pub struct BeadieJit {
    runtime_symbols: RuntimeSymbolTable,
    opt_level: &'static str,
}

impl BeadieJit {
    /// Build a fresh-backend adapter. `opt_level` is forwarded to
    /// [`CraneliftBackend::with_symbols_and_opt`] on every dispatch.
    pub fn new(runtime_symbols: RuntimeSymbolTable, opt_level: &'static str) -> Self {
        Self {
            runtime_symbols,
            opt_level,
        }
    }
}

/// Stage + finalize a fresh `CraneliftBackend` containing every loaded
/// module, then extract the target function pointer. Returns
/// `(ptr, backend)`; the caller is responsible for keeping `backend`
/// alive (via `Box::leak`) for the lifetime of `ptr`.
fn compile_into_fresh_backend(
    def: &BeadieFunctionDef,
    symbols: &[(String, usize)],
    opt_level: &str,
) -> Result<(usize, CraneliftBackend), String> {
    // Rebuild the `(name, ptr)` slice cranelift expects.
    let symbols_ref: Vec<(&str, *const u8)> = symbols
        .iter()
        .map(|(name, ptr)| (name.as_str(), *ptr as *const u8))
        .collect();
    let mut backend = CraneliftBackend::with_symbols_and_opt(opt_level, &symbols_ref)?;

    let modules = def.modules.read().unwrap();
    // Verify the target function is actually in some module before
    // committing to a multi-module compile — surfaces a useful error
    // on a stale dispatch instead of crashing inside the loop.
    if !modules
        .iter()
        .any(|m| m.functions.contains_key(&def.func_id))
    {
        return Err(format!(
            "BeadieJit: function {:?} not in any loaded module",
            def.func_id
        ));
    }
    for module in modules.iter() {
        backend.compile_module_without_finalize(module)?;
    }
    backend.finalize()?;
    let ptr = backend.get_function_ptr(def.func_id)? as usize;
    Ok((ptr, backend))
}

impl JitBackend for BeadieJit {
    type FunctionDef = BeadieFunctionDef;
    type Error = CompileError;

    /// Single-shot path: full multi-module compile + extract pointer +
    /// leak the backend. Used by `BackendAdapter::with_policy`.
    fn compile(&self, _bead: &Arc<Bead>, def: BeadieFunctionDef) -> Result<*mut (), Self::Error> {
        let (ptr, backend) =
            compile_into_fresh_backend(&def, &self.runtime_symbols, self.opt_level)
                .map_err(CompileError::new)?;
        // Keep the JIT memory backing `ptr` alive for the rest of the
        // process. Matches the legacy `compile_all_at_tier` pattern at
        // `tiered_backend.rs` ~line 2470.
        Box::leak(Box::new(backend));
        Ok(ptr as *mut ())
    }

    /// Batched path: compile + leak inside `compile_outcome`, return
    /// the pointer immediately via `CompileOutcome::Ready`. The bead
    /// transitions to `Compiled` as soon as the broker thread
    /// observes the outcome — no separate flush is needed for
    /// correctness (we always finalize before returning), so `flush`
    /// is a no-op.
    fn compile_outcome(
        &self,
        _bead: &Arc<Bead>,
        def: BeadieFunctionDef,
    ) -> Result<CompileOutcome, Self::Error> {
        let (ptr, backend) =
            compile_into_fresh_backend(&def, &self.runtime_symbols, self.opt_level)
                .map_err(CompileError::new)?;
        Box::leak(Box::new(backend));
        Ok(CompileOutcome::Ready(ptr as *mut ()))
    }

    /// No-op: every compile is self-contained (own backend, own
    /// finalize). The broker's batched-flush hook still fires but has
    /// no work to do.
    fn flush(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Build a batched [`BackendAdapter<BeadieJit>`] with the given runtime
/// symbol snapshot, at the cranelift opt-level corresponding to
/// rayzor's Standard tier.
pub fn build_batched_adapter(
    runtime_symbols: RuntimeSymbolTable,
    threshold: u32,
    capacity: usize,
    batch_limit: usize,
) -> Arc<BackendAdapter<BeadieJit>> {
    build_batched_adapter_at(
        runtime_symbols,
        super::tiered_backend::OptimizationTier::Standard.cranelift_opt_level(),
        threshold,
        capacity,
        batch_limit,
    )
}

/// Build a batched [`BackendAdapter<BeadieJit>`] with explicit
/// cranelift opt-level. Used by `TieredBackend` to construct one
/// adapter per tier (Standard at `"speed"`, Optimized at
/// `"speed_and_size"`).
pub fn build_batched_adapter_at(
    runtime_symbols: RuntimeSymbolTable,
    opt_level: &'static str,
    threshold: u32,
    capacity: usize,
    batch_limit: usize,
) -> Arc<BackendAdapter<BeadieJit>> {
    let jit = BeadieJit::new(runtime_symbols, opt_level);
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

    fn runtime_symbol_snapshot() -> RuntimeSymbolTable {
        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        Arc::new(
            symbols
                .iter()
                .map(|(n, p)| (n.to_string(), *p as usize))
                .collect(),
        )
    }

    /// The adapter constructs cleanly and the supplied runtime symbol
    /// table is adopted (held internally as the snapshot beadie will
    /// reuse on every compile).
    #[test]
    fn build_adapter_smoke() {
        let symbols = runtime_symbol_snapshot();
        let _adapter = build_batched_adapter(Arc::clone(&symbols), 1, 16, 4);
    }

    /// Step 6: beadie is unconditional. Constructing any
    /// `TieredBackend` always builds the Standard + Optimized
    /// adapters; there's no toggle to verify.
    #[test]
    fn tiered_backend_constructs_with_beadie_adapter() {
        use super::super::tiered_backend::{TieredBackend, TieredConfig};

        let backend = TieredBackend::new(TieredConfig::default()).expect("tiered backend");
        // Both calls return a real handle (Arc, no Option) — compile
        // would have failed if the adapter wasn't there.
        let _std_adapter = backend.beadie_adapter();
    }

    /// End-to-end: drive a real compile through beadie inside a live
    /// `TieredBackend`. The fresh-backend pattern means
    /// `bound.bead().compiled()` becomes `Some(ptr)` as soon as
    /// `on_invoke_outcome` returns (the compile runs synchronously on
    /// the broker thread and the outcome is `Ready`, not `Deferred`).
    #[test]
    fn tiered_backend_drives_real_compile_via_beadie() {
        use super::super::tiered_backend::{TieredBackend, TieredConfig};
        use beadie::BeadState;
        use std::time::{Duration, Instant};

        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        let mut cfg = TieredConfig::default();

        cfg.profile_config.warm_threshold = 1;

        let backend =
            TieredBackend::with_symbols(cfg, &symbols_ref).expect("tiered backend with beadie");
        let adapter = Arc::clone(backend.beadie_adapter());
        let (module, func_id) = build_add_module();
        let bound = adapter.register(std::ptr::null_mut(), None);

        let (_def, modules_handle) = BeadieFunctionDef::from_single_module(module, func_id);
        let outcome = adapter.on_invoke_outcome(&bound, move |_| BeadieFunctionDef {
            modules: modules_handle,
            func_id,
        });
        // First call before the threshold is met returns None.
        assert!(outcome.is_none());

        let deadline = Instant::now() + Duration::from_secs(5);
        while bound.bead().compiled().is_none() {
            if Instant::now() > deadline {
                panic!("beadie did not install compiled pointer within 5s");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(bound.bead().state(), BeadState::Compiled);

        let code = bound.bead().compiled().expect("compiled pointer");
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(20, 22), 42);
    }

    /// `ensure_beadie_bead` is idempotent. (Step 6: the disabled-mode
    /// error case is gone — beadie is unconditional.)
    #[test]
    fn ensure_beadie_bead_idempotence() {
        use super::super::tiered_backend::{TieredBackend, TieredConfig};
        use crate::tast::SymbolId;

        let backend = TieredBackend::new(TieredConfig::default()).expect("tiered backend");
        let dummy = IrFunctionId(SymbolId(7).into());

        assert!(!backend.has_beadie_bead(dummy));
        backend.ensure_beadie_bead(dummy).unwrap();
        assert!(backend.has_beadie_bead(dummy));
        backend.ensure_beadie_bead(dummy).unwrap(); // idempotent
        assert_eq!(backend.beadie_beads().lock().unwrap().len(), 1);
    }

    /// E2E for `record_call`: past warm threshold, Standard promotion
    /// is routed through beadie, the install happens under the
    /// `PromotionBarrier`, and the resulting pointer is callable.
    #[test]
    fn record_call_routes_standard_via_beadie_end_to_end() {
        use super::super::tiered_backend::{OptimizationTier, TieredBackend, TieredConfig};
        use std::time::{Duration, Instant};

        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        let mut cfg = TieredConfig::default();

        cfg.profile_config.warm_threshold = 1;
        cfg.profile_config.interpreter_threshold = 100;

        let mut backend =
            TieredBackend::with_symbols(cfg, &symbols_ref).expect("tiered backend with beadie");
        let (module, func_id) = build_add_module();
        backend.compile_module(module).expect("module loaded");

        backend.record_call(func_id);

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

        backend.record_call(func_id);

        assert_eq!(
            backend.function_tier(func_id),
            Some(OptimizationTier::Standard)
        );
        let installed_ptr = backend
            .jit_pointer(func_id)
            .expect("function_pointers missing entry after install");
        assert_eq!(installed_ptr, beadie_ptr as usize);

        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(beadie_ptr) };
        assert_eq!(f(19, 23), 42);
    }

    /// Phase B step 5: hot_threshold crossing routes Optimized
    /// promotion through beadie too. With `warm_threshold = 1` and
    /// `hot_threshold = 2`, the second `record_call` crosses
    /// `hot_threshold` and should route via the Optimized adapter.
    /// Asserts the function lands at `Optimized` tier with a pointer
    /// produced by beadie's `"speed_and_size"` opt-level compile.
    #[test]
    fn record_call_routes_optimized_via_beadie_end_to_end() {
        use super::super::tiered_backend::{OptimizationTier, TieredBackend, TieredConfig};
        use std::time::{Duration, Instant};

        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        let mut cfg = TieredConfig::default();

        cfg.profile_config.interpreter_threshold = 100; // skip Baseline
        cfg.profile_config.warm_threshold = 1;
        cfg.profile_config.hot_threshold = 2;
        cfg.profile_config.blazing_threshold = u64::MAX;

        let mut backend =
            TieredBackend::with_symbols(cfg, &symbols_ref).expect("tiered backend with beadie");
        let (module, func_id) = build_add_module();
        backend.compile_module(module).expect("module loaded");

        // First call: count 0→1, target = Standard, routes via beadie
        // Standard adapter.
        backend.record_call(func_id);
        // Second call: count 1→2, target = Optimized, routes via
        // beadie Optimized adapter.
        backend.record_call(func_id);

        // Wait for both compiles + at least one install.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            // Poll execute_function-style: see if any pointer landed.
            backend.record_call(func_id);
            if let Some(tier) = backend.function_tier(func_id) {
                if tier == OptimizationTier::Optimized {
                    break;
                }
            }
            if Instant::now() > deadline {
                panic!(
                    "Optimized tier not reached within 10s (last tier: {:?}, stats: {:?})",
                    backend.function_tier(func_id),
                    backend.beadie_stats()
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        assert_eq!(
            backend.function_tier(func_id),
            Some(OptimizationTier::Optimized)
        );
        let installed = backend
            .jit_pointer(func_id)
            .expect("function_pointers missing entry");
        let f: extern "C" fn(i64, i64) -> i64 =
            unsafe { std::mem::transmute(installed as *mut ()) };
        assert_eq!(f(40, 2), 42);

        // Stats sanity: at least 2 routes happened (Standard + Optimized),
        // at least 1 install landed.
        let stats = backend.beadie_stats();
        assert!(stats.routes_attempted >= 2, "stats: {:?}", stats);
        assert!(stats.installs >= 1, "stats: {:?}", stats);
    }

    /// Multi-function cross-call: `caller` invokes `callee` (both in
    /// the same module). Proves beadie's fresh-backend compile resolves
    /// internal cross-function references end-to-end. The
    /// single-module case is the minimum demonstration; the same
    /// machinery handles cross-*module* calls because every loaded
    /// module is compiled into the same fresh backend on every
    /// dispatch.
    #[test]
    fn beadie_resolves_cross_function_calls() {
        use super::super::tiered_backend::{TieredBackend, TieredConfig};
        use beadie::BeadState;
        use std::time::{Duration, Instant};

        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        // Build a module containing two functions: callee(x) = x * 2,
        // caller(x) = callee(x) + 1.
        let mut builder = MirBuilder::new("beadie_cross_call_test");
        let callee_id = builder
            .begin_function("callee")
            .param("x", IrType::I64)
            .returns(IrType::I64)
            .build();
        builder.set_current_function(callee_id);
        let callee_entry = builder.create_block("entry");
        builder.set_insert_point(callee_entry);
        let x = builder.get_param(0);
        let two = builder.const_i64(2);
        let doubled = builder.bin_op(BinaryOp::Mul, x, two);
        builder.ret(Some(doubled));

        let caller_id = builder
            .begin_function("caller")
            .param("x", IrType::I64)
            .returns(IrType::I64)
            .build();
        builder.set_current_function(caller_id);
        let caller_entry = builder.create_block("entry");
        builder.set_insert_point(caller_entry);
        let x = builder.get_param(0);
        let call_result = builder
            .call(callee_id, vec![x])
            .expect("call should produce a result for non-void return");
        let one = builder.const_i64(1);
        let final_result = builder.bin_op(BinaryOp::Add, call_result, one);
        builder.ret(Some(final_result));

        let module = builder.finish();

        let mut cfg = TieredConfig::default();

        cfg.profile_config.warm_threshold = 1;

        let backend =
            TieredBackend::with_symbols(cfg, &symbols_ref).expect("tiered backend with beadie");
        let adapter = Arc::clone(backend.beadie_adapter());
        let bound = adapter.register(std::ptr::null_mut(), None);

        let (_def, modules_handle) = BeadieFunctionDef::from_single_module(module, caller_id);
        let _ = adapter.on_invoke_outcome(&bound, move |_| BeadieFunctionDef {
            modules: modules_handle,
            func_id: caller_id,
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        while bound.bead().compiled().is_none() {
            if Instant::now() > deadline {
                panic!("beadie did not install compiled pointer within 5s");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(bound.bead().state(), BeadState::Compiled);

        let code = bound.bead().compiled().expect("compiled pointer");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };
        // caller(10) = callee(10) + 1 = (10 * 2) + 1 = 21
        assert_eq!(f(10), 21);
    }
}
