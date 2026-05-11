//! Integration smoke test: drive rayzor's Cranelift backend through
//! beadie's batched-compile API.
//!
//! What this proves:
//! - beadie's `JitBackend` trait fits rayzor's MIR + CraneliftBackend shape
//!   without changing rayzor's existing compile pipeline.
//! - `BackendAdapter::with_policy_batched` + `on_invoke_outcome` produce
//!   a usable native function pointer end-to-end.
//! - The batched flush hook maps cleanly onto rayzor's
//!   `compile_module_without_finalize` + `finalize` two-step.
//!
//! What this is *not*: a replacement for `TieredBackend`. Tiered policy
//! escalation, RTTI registration, stack-trace instrumentation, the
//! interpreter bailout protocol, etc. all stay where they are. This is
//! Phase A's spike — wire beadie in, drive a real compile, verify it
//! works. Phase B can begin replacing the hand-rolled counter + worker
//! once the shape is validated.

#![allow(clippy::missing_safety_doc)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beadie::{
    BackendAdapter, Bead, BeadState, BoundBead, CompileError, CompileOutcome, JitBackend,
    ThresholdPolicy,
};
use compiler::codegen::CraneliftBackend;
use compiler::ir::mir_builder::MirBuilder;
use compiler::ir::{BinaryOp, IrFunctionId, IrModule, IrType};

// ────────────────────────────────────────────────────────────────────────────
// JitBackend impl wrapping rayzor's CraneliftBackend
// ────────────────────────────────────────────────────────────────────────────
//
// The `FunctionDef` is the IrModule + the target IrFunctionId. rayzor's
// CraneliftBackend compiles whole modules at once and looks functions up by
// IrFunctionId, so the natural beadie integration registers one bead per
// (module, func_id) pair.
//
// For the smoke test we keep each module single-function, so staging one
// module = staging one function. A real adoption would stage a whole
// stdlib module behind one bead (or one bead per function with a shared
// module that's compiled lazily on the first stage).

/// A single function the broker will JIT — an IrModule + the function id
/// we expect to extract a pointer for.
struct RayzorDef {
    module: Arc<IrModule>,
    func_id: IrFunctionId,
}

/// JitBackend adaptor over rayzor's CraneliftBackend.
///
/// The backend is wrapped in `Arc<Mutex<...>>` so beadie's `&self`
/// `compile` / `flush` methods can mutate it. compile_outcome stages a
/// module via `compile_module_without_finalize` and returns Deferred with
/// a resolver that reads `get_function_ptr` after the batch's `finalize()`
/// runs.
struct RayzorJit {
    backend: Arc<Mutex<CraneliftBackend>>,
}

impl JitBackend for RayzorJit {
    type FunctionDef = RayzorDef;
    type Error = CompileError;

    /// Single-shot: stage + finalize + read pointer. Use when batching
    /// isn't beneficial (e.g. ad-hoc tier promotion of one function).
    fn compile(&self, _bead: &Arc<Bead>, def: RayzorDef) -> Result<*mut (), Self::Error> {
        let mut be = self.backend.lock().unwrap();
        be.compile_module_without_finalize(&def.module)
            .map_err(CompileError::new)?;
        be.finalize().map_err(CompileError::new)?;
        let ptr = be
            .get_function_ptr(def.func_id)
            .map_err(CompileError::new)?;
        Ok(ptr as *mut ())
    }

    /// Batched: stage now (compile_module_without_finalize), defer the
    /// pointer read until the broker's flush() runs finalize() once.
    fn compile_outcome(
        &self,
        _bead: &Arc<Bead>,
        def: RayzorDef,
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

// ────────────────────────────────────────────────────────────────────────────
// MIR helpers — build a tiny `(i64, i64) -> i64` function
// ────────────────────────────────────────────────────────────────────────────

/// Build an IR module containing one function: `fn add(a: i64, b: i64) -> i64`.
/// Returns the module and the IrFunctionId of `add`.
fn build_add_module() -> (Arc<IrModule>, IrFunctionId) {
    let mut builder = MirBuilder::new("beadie_smoke");
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

    let module = builder.finish();
    (Arc::new(module), func_id)
}

// ────────────────────────────────────────────────────────────────────────────
// Test phases
// ────────────────────────────────────────────────────────────────────────────

fn wait_compiled(bead: &Arc<Bead>, timeout: Duration, label: &str) {
    let deadline = Instant::now() + timeout;
    while bead.compiled().is_none() {
        if Instant::now() > deadline {
            panic!("{label}: broker did not install within {timeout:?}");
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Phase 1: single-shot compile via the default `compile()` path. Validates
/// that beadie's existing flow works against rayzor's backend.
fn run_single_shot(jit: Arc<RayzorJit>) -> i64 {
    let policy = ThresholdPolicy::new(1); // Promote on first call.
    let adapter: BackendAdapter<RayzorJit> = BackendAdapter::from_arc_with_policy(jit, policy);

    let (module, func_id) = build_add_module();
    let bound: BoundBead<RayzorJit> = adapter.register(std::ptr::null_mut(), None);

    // First call triggers promotion (threshold=1). Returns None — the
    // compile runs on the broker thread.
    let module_arc = Arc::clone(&module);
    let result = adapter.on_invoke(&bound, move |_| RayzorDef {
        module: module_arc,
        func_id,
    });
    assert!(
        result.is_none(),
        "first call before compile must return None"
    );

    wait_compiled(bound.bead(), Duration::from_secs(5), "single_shot");

    // Now the bead has a pointer. Call it.
    let code = bound.bead().compiled().expect("compiled pointer");
    let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
    f(7, 35)
}

/// Phase 2: batched compile via `with_policy_batched` + `compile_outcome`.
/// Same expected result; just exercises the staged-then-flush path.
fn run_batched(jit: Arc<RayzorJit>) -> i64 {
    let policy = ThresholdPolicy::new(1);
    let adapter: BackendAdapter<RayzorJit> = BackendAdapter::from_arc_with_policy_batched(
        jit, policy, /*capacity=*/ 16, /*batch=*/ 4,
    );

    let (module, func_id) = build_add_module();
    let bound: BoundBead<RayzorJit> = adapter.register(std::ptr::null_mut(), None);

    let module_arc = Arc::clone(&module);
    let result = adapter.on_invoke_outcome(&bound, move |_| RayzorDef {
        module: module_arc,
        func_id,
    });
    assert!(
        result.is_none(),
        "first call before compile must return None"
    );

    wait_compiled(bound.bead(), Duration::from_secs(5), "batched");

    let code = bound.bead().compiled().expect("compiled pointer");
    let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
    f(100, 23)
}

// ────────────────────────────────────────────────────────────────────────────
// main
// ────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("== beadie ↔ rayzor smoke test ==");

    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    // Single-shot phase.
    {
        let backend = CraneliftBackend::with_symbols(&symbols_ref).expect("cranelift backend");
        let jit = Arc::new(RayzorJit {
            backend: Arc::new(Mutex::new(backend)),
        });
        let answer = run_single_shot(jit);
        assert_eq!(
            answer, 42,
            "single_shot: add(7, 35) expected 42, got {answer}"
        );
        println!("  [OK] single-shot: add(7, 35) = {answer}");
    }

    // Batched phase — a fresh backend so the JITModule starts clean.
    {
        let backend = CraneliftBackend::with_symbols(&symbols_ref).expect("cranelift backend");
        let jit = Arc::new(RayzorJit {
            backend: Arc::new(Mutex::new(backend)),
        });
        let answer = run_batched(jit);
        assert_eq!(
            answer, 123,
            "batched: add(100, 23) expected 123, got {answer}"
        );
        println!("  [OK] batched:     add(100, 23) = {answer}");
    }

    // Bead-state sanity — both paths should drive the bead to Compiled.
    {
        let backend = CraneliftBackend::with_symbols(&symbols_ref).expect("cranelift backend");
        let jit = RayzorJit {
            backend: Arc::new(Mutex::new(backend)),
        };
        let adapter: BackendAdapter<RayzorJit> =
            BackendAdapter::with_policy_batched(jit, ThresholdPolicy::new(1), 8, 4);
        let (module, func_id) = build_add_module();
        let bound = adapter.register(std::ptr::null_mut(), None);
        let module_arc = Arc::clone(&module);
        let _ = adapter.on_invoke_outcome(&bound, move |_| RayzorDef {
            module: module_arc,
            func_id,
        });
        wait_compiled(bound.bead(), Duration::from_secs(5), "state_check");
        assert_eq!(bound.bead().state(), BeadState::Compiled);
        println!("  [OK] bead state reached Compiled");
    }

    println!("== All smoke checks passed ==");
}
