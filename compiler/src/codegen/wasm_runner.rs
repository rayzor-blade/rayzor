//! WASM Runner — executes compiled WASM binaries via embedded wasmtime.
//!
//! Used by `rayzor run --wasm` to run WASM programs without an external
//! wasmtime installation. Provides WASI P1 imports for stdout, filesystem,
//! environment, and clocks, plus host implementations for haxe.io.Bytes.
//!
//! Capability side-modules (e.g. the nue Q8 KV cache) load via the standard
//! wasm dynamic-linking ABI (dylink.0 + GOT): the runner instantiates a PIC
//! side-module into the same Store, assigns it `__memory_base`/`__table_base`,
//! resolves its GOT, and trampolines the guest's capability imports to it. See
//! `parse_side_dylink` + the side-module load block in `run_wasm_with_args`.

/// GOT resolution data parsed statically from a PIC dylink.0 side-module.
///
/// `GOT.func.<sym>` / `GOT.mem.<sym>` are imported globals (module
/// `"GOT.func"`/`"GOT.mem"`, name = the mangled symbol). The side also exports
/// each referenced function (so symbol → func index → table slot) and
/// re-exports each GOT.mem data symbol as a global whose const init is the
/// symbol's byte offset within the side's data image. The loader fills the
/// imported GOT globals with `__table_base + slot` / `__memory_base + offset`.
#[cfg(feature = "wasm-runtime")]
#[derive(Default)]
struct SideDylink {
    /// Size in bytes of the side's data image (from the `dylink.0` mem-info).
    /// The loader reserves this much for `__memory_base`.
    mem_size: u32,
    /// GOT.func symbol → table slot (relative to `__table_base`).
    got_func_slot: std::collections::HashMap<String, i32>,
    /// GOT.mem symbol → byte offset in the data image (relative to `__memory_base`).
    got_mem_offset: std::collections::HashMap<String, i32>,
}

/// Coerce a wasm value to a target ValType at the trampoline ABI boundary.
/// The guest emits Haxe `Int` (i32) for the Q8 capability args while the side
/// module's exports use `i64` (the manifest's `I64`), and the i64 handle it
/// returns is narrowed back to the guest's i32 `Ptr`. Pointers/handles live in
/// 32-bit wasm linear memory so the i64↔i32 narrowing is lossless here.
#[cfg(feature = "wasm-runtime")]
fn coerce_val(v: &wasmtime::Val, target: &wasmtime::ValType) -> wasmtime::Val {
    use wasmtime::{Val, ValType};
    match (v, target) {
        (Val::I32(x), ValType::I64) => Val::I64(*x as i64),
        (Val::I64(x), ValType::I32) => Val::I32(*x as i32),
        (Val::F32(x), ValType::F64) => Val::F64((f32::from_bits(*x) as f64).to_bits()),
        (Val::F64(x), ValType::F32) => Val::F32((f64::from_bits(*x) as f32).to_bits()),
        (other, _) => other.clone(),
    }
}

/// Read an unsigned LEB128 from `buf` at `*pos`, advancing `*pos`.
#[cfg(feature = "wasm-runtime")]
fn read_uleb32(buf: &[u8], pos: &mut usize) -> u32 {
    let mut result: u32 = 0;
    let mut shift = 0;
    while *pos < buf.len() {
        let b = buf[*pos];
        *pos += 1;
        result |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

#[cfg(feature = "wasm-runtime")]
fn parse_side_dylink(bytes: &[u8]) -> Result<SideDylink, String> {
    use std::collections::HashMap;
    use wasmparser::{ElementItems, ElementKind, ExternalKind, Operator, Parser, Payload, TypeRef};

    let mut mem_size: u32 = 0;
    let mut got_func_syms: Vec<String> = Vec::new();
    let mut got_mem_syms: Vec<String> = Vec::new();
    let mut num_imported_globals: u32 = 0;
    let mut export_func_idx: HashMap<String, u32> = HashMap::new();
    let mut export_global_idx: HashMap<String, u32> = HashMap::new();
    // full global index → i32 const init value (defined globals only).
    let mut global_const: HashMap<u32, i32> = HashMap::new();
    // func index → table slot (from active element segments; base = __table_base = 0).
    let mut func_slot: HashMap<u32, i32> = HashMap::new();

    for payload in Parser::new(0).parse_all(bytes) {
        match payload.map_err(|e| format!("side dylink parse: {e}"))? {
            Payload::ImportSection(reader) => {
                for group in reader {
                    let group = group.map_err(|e| format!("side import group: {e}"))?;
                    for item in group {
                        let (_off, imp) = item.map_err(|e| format!("side import: {e}"))?;
                        if matches!(imp.ty, TypeRef::Global(_)) {
                            match imp.module {
                                "GOT.func" => got_func_syms.push(imp.name.to_string()),
                                "GOT.mem" => got_mem_syms.push(imp.name.to_string()),
                                _ => {}
                            }
                            num_imported_globals += 1;
                        }
                    }
                }
            }
            Payload::CustomSection(c) if c.name() == "dylink.0" => {
                // Subsections: <id:u8> <size:uleb> <payload>. mem-info (id 1)
                // payload = mem_size, mem_align, table_size, table_align (uleb).
                let d = c.data();
                let mut p = 0usize;
                while p < d.len() {
                    let id = d[p];
                    p += 1;
                    let size = read_uleb32(d, &mut p) as usize;
                    let sub_end = (p + size).min(d.len());
                    if id == 1 {
                        mem_size = read_uleb32(d, &mut p);
                    }
                    p = sub_end;
                }
            }
            Payload::ExportSection(reader) => {
                for exp in reader {
                    let exp = exp.map_err(|e| format!("side export: {e}"))?;
                    match exp.kind {
                        ExternalKind::Func => {
                            export_func_idx.insert(exp.name.to_string(), exp.index);
                        }
                        ExternalKind::Global => {
                            export_global_idx.insert(exp.name.to_string(), exp.index);
                        }
                        _ => {}
                    }
                }
            }
            Payload::GlobalSection(reader) => {
                for (i, g) in reader.into_iter().enumerate() {
                    let g = g.map_err(|e| format!("side global: {e}"))?;
                    // Init is `i32.const off` or `global.get __memory_base; i32.const off; i32.add`;
                    // the i32.const operand is the in-image offset either way.
                    let mut ops = g.init_expr.get_operators_reader();
                    let mut off: i32 = 0;
                    while let Ok(op) = ops.read() {
                        match op {
                            Operator::I32Const { value } => off = value,
                            Operator::End => break,
                            _ => {}
                        }
                    }
                    global_const.insert(num_imported_globals + i as u32, off);
                }
            }
            Payload::ElementSection(reader) => {
                for elem in reader {
                    let elem = elem.map_err(|e| format!("side elem: {e}"))?;
                    if let ElementKind::Active { offset_expr, .. } = elem.kind {
                        // Offset is `global.get __table_base` (base 0) or an i32.const.
                        let mut base: i32 = 0;
                        let mut ops = offset_expr.get_operators_reader();
                        while let Ok(op) = ops.read() {
                            match op {
                                Operator::I32Const { value } => base = value,
                                Operator::End => break,
                                _ => {}
                            }
                        }
                        match elem.items {
                            // `(elem ... func $a $b ...)` shorthand.
                            ElementItems::Functions(funcs) => {
                                let mut slot = base;
                                for f in funcs {
                                    let f = f.map_err(|e| format!("side elem item: {e}"))?;
                                    func_slot.insert(f, slot);
                                    slot += 1;
                                }
                            }
                            // PIC modules emit `(elem ... funcref (ref.func $a) ...)`.
                            ElementItems::Expressions(_ty, exprs) => {
                                let mut slot = base;
                                for ex in exprs {
                                    let ex = ex.map_err(|e| format!("side elem expr: {e}"))?;
                                    let mut ops = ex.get_operators_reader();
                                    while let Ok(op) = ops.read() {
                                        match op {
                                            Operator::RefFunc { function_index } => {
                                                func_slot.insert(function_index, slot);
                                            }
                                            Operator::End => break,
                                            _ => {}
                                        }
                                    }
                                    slot += 1;
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // GOT.func entries that resolve to a table slot get it; the rest are dead
    // fn-pointers (core::fmt/Debug machinery whose addresses are only stored in
    // unreachable vtables — LLVM emits the GOT entry but never table-places the
    // func). Default those to slot 0: __wasm_apply_data_relocs writes the value
    // into dead .data, never dereferenced. `unresolved_func` counts them so a
    // surprising trap can be traced back here.
    let mut got_func_slot = HashMap::new();
    let mut unresolved_func = 0usize;
    for sym in got_func_syms {
        let slot = export_func_idx
            .get(&sym)
            .and_then(|fidx| func_slot.get(fidx))
            .copied();
        if slot.is_none() {
            unresolved_func += 1;
        }
        got_func_slot.insert(sym, slot.unwrap_or(0));
    }
    let mut got_mem_offset = HashMap::new();
    let mut unresolved_mem = 0usize;
    for sym in got_mem_syms {
        let off = export_global_idx
            .get(&sym)
            .and_then(|gidx| global_const.get(gidx))
            .copied();
        if off.is_none() {
            unresolved_mem += 1;
        }
        got_mem_offset.insert(sym, off.unwrap_or(0));
    }
    if unresolved_func > 0 || unresolved_mem > 0 {
        eprintln!(
            "[dylink] {unresolved_func}/{} GOT.func + {unresolved_mem}/{} GOT.mem unresolved (defaulted to 0 — dead fn-ptrs/data)",
            got_func_slot.len(),
            got_mem_offset.len()
        );
    }

    Ok(SideDylink {
        mem_size,
        got_func_slot,
        got_mem_offset,
    })
}

#[cfg(all(test, feature = "wasm-runtime"))]
mod dylink_tests {
    #[test]
    fn parses_q8_side_module_got() {
        // Validates the dylink.0 GOT parser against the real nue-plugins PIC
        // side-module. Skips if the artifact isn't built (build it with
        // nue-plugins' wasm32-wasip1-threads-pic recipe).
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../target/wasm32-wasip1-threads-pic/release/nue_plugins.wasm"
        );
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skip: side-module not built at {path}");
            return;
        };
        let d = super::parse_side_dylink(&bytes).expect("parse side dylink");
        // 24 GOT.func + 12 GOT.mem expected from the current build.
        assert!(
            !d.got_func_slot.is_empty(),
            "expected GOT.func entries, got none"
        );
        assert!(
            !d.got_mem_offset.is_empty(),
            "expected GOT.mem entries, got none"
        );
        // Slots must be within the table (size 53); offsets within the data
        // image (size 49440).
        for (sym, &slot) in &d.got_func_slot {
            assert!(
                (0..53).contains(&slot),
                "GOT.func {sym} slot {slot} out of table"
            );
        }
        for (sym, &off) in &d.got_mem_offset {
            assert!(
                (0..49440).contains(&off),
                "GOT.mem {sym} offset {off} out of data image"
            );
        }
        eprintln!(
            "parsed {} GOT.func + {} GOT.mem; sample slot={:?} off={:?}",
            d.got_func_slot.len(),
            d.got_mem_offset.len(),
            d.got_func_slot.values().next(),
            d.got_mem_offset.values().next()
        );
    }
}

/// Locate the nue Q8 capability PIC side-module (phase 1: a direct `.wasm`
/// path). `RAYZOR_Q8_SIDE_MODULE` overrides; otherwise the default build
/// location relative to the workspace/cwd. (Phase 2 sources this from an
/// `.rpkg`; phase 3 async-fetches it in the browser.)
#[cfg(feature = "wasm-runtime")]
fn find_wasm_side_module() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(p) = std::env::var_os("RAYZOR_Q8_SIDE_MODULE") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // Relative to the rayzor executable (`<root>/target/release/rayzor` →
    // `<root>/target/wasm32-wasip1-threads-pic/release/nue_plugins.wasm`), so it
    // resolves regardless of the process CWD — `rayzor run` cd's into the
    // project dir, which broke the bare CWD-relative paths below.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(target_dir) = exe.parent().and_then(|p| p.parent()) {
            let p = target_dir.join("wasm32-wasip1-threads-pic/release/nue_plugins.wasm");
            if p.exists() {
                return Some(p);
            }
        }
    }
    for c in [
        "target/wasm32-wasip1-threads-pic/release/nue_plugins.wasm",
        "../target/wasm32-wasip1-threads-pic/release/nue_plugins.wasm",
    ] {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[cfg(feature = "wasm-runtime")]
pub fn run_wasm(wasm_bytes: &[u8]) -> Result<(), String> {
    run_wasm_with_args(wasm_bytes, &[])
}

/// Variant of `run_wasm` that threads CLI tail args (everything after `--` on
/// the rayzor invocation) through to the sandbox. The args become visible to
/// the Haxe program via `Sys.args()`, which lowers to a `haxe_sys_args` import
/// the runner registers as a host stub.
#[cfg(feature = "wasm-runtime")]
pub fn run_wasm_with_args(wasm_bytes: &[u8], program_args: &[String]) -> Result<(), String> {
    use std::collections::{BTreeMap, BTreeSet};
    use wasmtime::*;

    // -- Host state accessible from all host functions via Caller::data_mut() --
    struct WasmState {
        wasi_ctx: wasi_common::p1::WasiP1Ctx,
        bytes_handles: BTreeMap<i32, Vec<u8>>,
        next_bytes_id: i32,
        /// EReg handle table: handle_id → ERegState
        ereg_handles: BTreeMap<i32, ERegState>,
        next_ereg_id: i32,
        /// Mutex handle table: handle_id → MutexState
        mutex_handles: BTreeMap<i32, MutexState>,
        next_mutex_id: i32,
        /// Tensor handle table: handle_id → TensorState
        tensor_handles: BTreeMap<i32, TensorState>,
        next_tensor_id: i32,
        /// Thread handle table: handle id → the completion slot a worker fills.
        /// Real OS threads run guest closures on their own Store+Instance
        /// (sharing `shared_memory`) — see `worker_pool`.
        thread_handles: BTreeMap<i32, ThreadState>,
        next_thread_id: i32,
        /// Persistent pool of worker threads. Each owns one Store+Instance of
        /// the module (instantiated once, sharing `shared_memory`) plus one
        /// private stack, then runs dispatched closures in a loop. Built lazily
        /// on the first `rayzor_thread_spawn`. Amortizing the (expensive)
        /// per-thread instantiation across every spawn is critical for the
        /// in-guest parallel matmul, which spawns N bands per call and hundreds
        /// of calls per token.
        worker_pool: Option<WorkerPool>,
        /// wgpu device + queue for native GPU compute (Metal/Vulkan/DX12)
        wgpu_ctx: Option<WgpuComputeCtx>,
        /// Host-side bump allocator: allocates downward from top of WASM memory.
        /// Used to write DynamicValue return structs into WASM linear memory.
        host_alloc_ptr: u32,
        /// Low watermark protecting the host-allocation region from being
        /// reused after a grow.
        host_alloc_low_bound: u32,
        /// The shared linear memory we defined as `env.memory`. Kept here
        /// so host functions can read/write it directly when `caller
        /// .get_export("memory")` doesn't surface the import (wasmtime
        /// doesn't expose imported SharedMemory via caller exports).
        shared_memory: Option<wasmtime::SharedMemory>,
        /// CLI tail args (everything after `--` on the rayzor command line).
        /// Surfaced to wasm via the `haxe_sys_args` host stub.
        program_args: Vec<String>,
        /// Haxe StringMap handle table. Each handle is a small integer the wasm
        /// program holds; the host stores the actual `BTreeMap<String, i64>` so
        /// every set/get/exists call goes through the host stubs. Values are
        /// i64 because Haxe StringMap is generic over `V` and the compiler
        /// emits all calls through the i64-stride MIR slot.
        stringmap_handles: BTreeMap<i32, BTreeMap<String, i64>>,
        next_stringmap_id: i32,
        /// The dynamically-loaded capability side-module (e.g. the nue Q8 KV
        /// cache), instantiated into this same Store via the dylink.0 ABI.
        /// `None` until the side-module load block runs (or if no capability
        /// imports are present). The Q8 host trampolines dispatch to it.
        /// `Instance` is a `Copy` handle, valid for the life of the Store.
        side_instance: Option<wasmtime::Instance>,
        /// Engine + compiled module, cloned into worker threads so a real
        /// `rayzor_thread_spawn` can instantiate the module on its own Store
        /// (sharing `shared_memory`) and run the closure in parallel. Both are
        /// `Arc`-backed (`Clone` + `Send` + `Sync`).
        engine: wasmtime::Engine,
        module: wasmtime::Module,
    }

    /// A spawned closure's completion slot. The dispatching (main) thread holds
    /// an `Arc` via `ThreadState`; the worker fills `done` and notifies `cv`.
    struct JobSlot {
        done: std::sync::Mutex<Option<i64>>,
        cv: std::sync::Condvar,
    }

    /// A unit of work dispatched to a pool worker: call indirect-table entry
    /// `fn_idx` with `env_ptr`, then publish the result into `slot`.
    struct Job {
        fn_idx: u32,
        env_ptr: i32,
        slot: std::sync::Arc<JobSlot>,
    }

    /// Persistent worker threads, each owning its own module instance + stack.
    struct WorkerPool {
        senders: Vec<std::sync::mpsc::Sender<Job>>,
        /// Round-robin dispatch cursor (mutated only on the main thread).
        next: usize,
    }

    struct ThreadState {
        /// Completion slot the worker fills with the closure's return value.
        slot: std::sync::Arc<JobSlot>,
        /// Cached result after the first join, so repeated joins/is_finished
        /// calls keep working.
        result: Option<i64>,
    }

    /// wgpu device + queue + compiled compute pipelines.
    /// Created once at startup, used by tensor host functions.
    struct WgpuComputeCtx {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl WgpuComputeCtx {
        fn new() -> Option<Self> {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))?;
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rayzor_wasmtime_compute"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            ))
            .ok()?;
            Some(WgpuComputeCtx { device, queue })
        }

        /// Run a matmul A (MxK) @ B (KxN) -> C (MxN) on the GPU.
        /// Uploads inputs, dispatches compute, downloads result. Synchronous.
        fn matmul(&self, a: &[f64], b: &[f64], m: u32, k: u32, n: u32) -> Vec<f64> {
            // Convert f64 -> f32 (wgpu doesn't support f64 in shaders by default)
            let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
            let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
            let a_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(a_f32.as_ptr() as *const u8, a_f32.len() * 4) };
            let b_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(b_f32.as_ptr() as *const u8, b_f32.len() * 4) };

            let a_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_a"),
                size: a_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let b_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_b"),
                size: b_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let c_size = (m * n * 4) as u64;
            let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_c"),
                size: c_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_read"),
                size: c_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dims_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("matmul_dims"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.queue.write_buffer(&a_buf, 0, a_bytes);
            self.queue.write_buffer(&b_buf, 0, b_bytes);
            let dims = [m, k, n, 0u32];
            let dims_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(dims.as_ptr() as *const u8, 16) };
            self.queue.write_buffer(&dims_buf, 0, dims_bytes);

            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    "struct Dims { M: u32, K: u32, N: u32, _pad: u32 };\n\
                     @group(0) @binding(0) var<uniform> dims: Dims;\n\
                     @group(0) @binding(1) var<storage, read> a: array<f32>;\n\
                     @group(0) @binding(2) var<storage, read> b: array<f32>;\n\
                     @group(0) @binding(3) var<storage, read_write> c: array<f32>;\n\
                     @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                       let row = gid.x; let col = gid.y;\n\
                       if (row >= dims.M || col >= dims.N) { return; }\n\
                       var s: f32 = 0.0;\n\
                       for (var p: u32 = 0u; p < dims.K; p = p + 1u) {\n\
                         s = s + a[row * dims.K + p] * b[p * dims.N + col];\n\
                       }\n\
                       c[row * dims.N + col] = s;\n\
                     }".into()
                ),
            });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("matmul_pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dims_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: c_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups((m + 7) / 8, (n + 7) / 8, 1);
            }
            encoder.copy_buffer_to_buffer(&c_buf, 0, &read_buf, 0, c_size);
            self.queue.submit(std::iter::once(encoder.finish()));

            let slice = read_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            let f32_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, (m * n) as usize)
            };
            let result: Vec<f64> = f32_slice.iter().map(|&x| x as f64).collect();
            drop(data);
            read_buf.unmap();
            result
        }
    }

    struct TensorState {
        data: Vec<f64>,
        shape: Vec<i32>,
    }

    struct MutexState {
        locked: bool,
        value: i32, // stored value (for guard_get)
    }

    struct ERegState {
        pattern: String,
        flags: String,
        regex: regex::Regex,
        /// Last input string (set by match/matchSub)
        last_input: Option<String>,
        /// Last match result
        last_match: Option<regex::Match<'static>>,
        /// Capture groups from last match (owned strings)
        last_captures: Vec<Option<String>>,
        /// Positions for matchedLeft/Right
        match_start: usize,
        match_end: usize,
    }

    fn val_i32(v: &Val) -> i32 {
        match v {
            Val::I32(x) => *x,
            Val::I64(x) => *x as i32,
            _ => 0,
        }
    }

    fn val_i64(v: &Val) -> i64 {
        match v {
            Val::I64(x) => *x,
            Val::I32(x) => *x as i64,
            _ => 0,
        }
    }

    fn val_f32(v: &Val) -> f32 {
        match v {
            Val::F32(bits) => f32::from_bits(*bits),
            Val::F64(bits) => f64::from_bits(*bits) as f32,
            _ => 0.0,
        }
    }

    fn val_f64(v: &Val) -> f64 {
        match v {
            Val::F64(bits) => f64::from_bits(*bits),
            Val::F32(bits) => f32::from_bits(*bits) as f64,
            _ => 0.0,
        }
    }

    fn wasm_addr(raw: i32) -> u32 {
        raw as u32
    }

    fn wasm_addr_usize(raw: i32) -> usize {
        wasm_addr(raw) as usize
    }

    /// Host-side parallel quant matmul over a wasm linear-memory region.
    ///
    /// The guest (`runtime-wasm` `rayzor_tensor_matmul_qt_t_f32_threaded`)
    /// allocates `y`, then hands us raw wasm byte-offsets for `x` (F32
    /// activation `[batch, k]`), the Q4_K_M/Q6_K weight blocks `[n, k]`, and
    /// `y` (`[batch, n]`). We prequantise each `x` row to Q8_K natively, then
    /// fan the `n` output columns across the native worker pool — each band
    /// runs `vec_dot_q*_K_q8_K` (the same kernels the guest uses, but native
    /// and threaded) and writes a disjoint slice of `y`.
    ///
    /// Soundness: the wasm memory is a fixed mmap (`memory_may_move(false)`),
    /// so `base` is stable for the whole call; the single guest thread is
    /// blocked in this host import, so no concurrent guest mutation occurs;
    /// worker bands write disjoint `y` columns and only read-share the weights
    /// and the prequantised `x`. Pointers cross the `'static + Send + Sync`
    /// `parallel_rows` boundary as `usize` and are reconstructed inside.
    ///
    /// Returns 1 when handled, 0 when the shape/scheme isn't supported (the
    /// guest then falls back to its sequential path).
    ///
    /// # Safety
    /// `base + *_off` must reference live wasm memory of the implied extents.
    #[allow(clippy::too_many_arguments)]
    unsafe fn host_parallel_qmatmul(
        base: usize,
        x_data_off: usize,
        x_stride0_elems: usize,
        batch: usize,
        k: usize,
        qt_data_off: usize,
        scheme: i32,
        n: usize,
        y_data_off: usize,
        guest_threads: i32,
    ) -> i32 {
        use rayzor_runtime_core::quant::matmul::prepare_x_q8k_blocks_into;
        use rayzor_runtime_core::quant::q6_k::vec_dot_q6_K_q8_K;
        use rayzor_runtime_core::quant::types::{
            Q4KMBlock, Q8KBlock, Q4_K_M_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QSCHEME_Q4_K_M, QSCHEME_Q6_K,
        };
        // Q4_K_M dot: native NEON SDOT (the llama.cpp-ported kernel, the host's
        // fastest) when the build has dotprod, else the portable scalar
        // dispatcher. Both take the same `Q8KBlock` activation we prequant
        // below, so the prequant is identical regardless of path.
        #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
        use rayzor_runtime_core::quant::q4_k_m::vec_dot_q4_K_q8_K;
        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
        use rayzor_runtime_core::quant::sdot::dot_q4_k_q8_kblock_fast;

        const BLOCK_SIZE: usize = 256;
        if k == 0 || n == 0 || batch == 0 || k % BLOCK_SIZE != 0 {
            return 0;
        }
        let block_bytes = match scheme as u8 {
            QSCHEME_Q4_K_M => Q4_K_M_BLOCK_BYTES,
            QSCHEME_Q6_K => Q6_K_BLOCK_BYTES,
            _ => return 0,
        };
        let blocks_per_row = k / BLOCK_SIZE;
        // Mirror native dispatch: positive guest hint wins (clamped), else auto.
        let threads = if guest_threads > 1 {
            (guest_threads as usize).min(64)
        } else {
            rayzor_runtime::worker_pool::auto_kernel_threads()
        };
        let qt_base = base + qt_data_off;
        let y_base = base + y_data_off;

        // Pre-quantise ALL batch rows once, then a SINGLE parallel_rows over
        // the output columns whose worker dots each weight row against every
        // batch row (mirrors the native batched kernel
        // qmatmul_chunk_impl_sdot_q4km_batch). For batch>1 (prompt prefill)
        // this collapses `batch` separate worker-pool fan-outs per matmul into
        // one, amortising dispatch/join overhead across the whole prompt — the
        // reason native prefill is faster per-token than its decode. batch==1
        // (decode) is unchanged.
        let mut x_q8k_all: Vec<Q8KBlock> = Vec::with_capacity(batch * blocks_per_row);
        let mut tmp: Vec<Q8KBlock> = Vec::with_capacity(blocks_per_row);
        for bch in 0..batch {
            let x_ptr = (base + x_data_off + bch * x_stride0_elems * 4) as *const f32;
            prepare_x_q8k_blocks_into(x_ptr, k, &mut tmp);
            x_q8k_all.extend_from_slice(&tmp[..blocks_per_row]);
        }
        let xq_ptr = x_q8k_all.as_ptr() as usize;
        rayzor_runtime::worker_pool::global().parallel_rows(n, threads, move |lo, hi| {
            let xq = xq_ptr as *const Q8KBlock;
            for n_idx in lo..hi {
                let row_ptr = (qt_base + n_idx * blocks_per_row * block_bytes) as *const u8;
                for bch in 0..batch {
                    let xq_row = unsafe { xq.add(bch * blocks_per_row) };
                    let mut sum = 0.0f32;
                    for blk in 0..blocks_per_row {
                        let bp = unsafe { row_ptr.add(blk * block_bytes) };
                        let xb = unsafe { &*xq_row.add(blk) };
                        sum += if scheme as u8 == QSCHEME_Q4_K_M {
                            let w = unsafe { &*(bp as *const Q4KMBlock) };
                            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
                            let d = unsafe { dot_q4_k_q8_kblock_fast(w, xb) };
                            #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
                            let d = vec_dot_q4_K_q8_K(w, xb);
                            d
                        } else {
                            unsafe { vec_dot_q6_K_q8_K(bp, xb) }
                        };
                    }
                    unsafe { *((y_base + bch * n * 4 + n_idx * 4) as *mut f32) = sum };
                }
            }
        });
        1
    }

    /// Host-parallel Q8_0 flash attention (decode, seq_q == 1). Mirrors the
    /// serial side-module kernel `rayzor_tensor_flash_attn_decode_q8` but bands
    /// the INDEPENDENT `kv_head`s across the native worker pool — the same pool
    /// the matmul uses. Unlike the in-guest wasm workers (no TLS → unsafe
    /// malloc), these are real OS threads, so per-band `Vec` scratch is safe.
    /// The per-`kv_head` compute is copied verbatim (same NEON helpers, same
    /// `libm::expf` as the wasm guest's `fmath::exp`, same block-by-block
    /// reduction order), and kv_heads write disjoint output rows, so the result
    /// is bit-identical to the serial path. Returns 1 on success, 0 to fall
    /// back to the guest serial kernel. Reads K/V/Q/out from wasm linear memory
    /// at the given offsets (`base` is the pinned mmap base).
    #[allow(clippy::too_many_arguments)]
    unsafe fn host_parallel_flash_q8(
        base: usize,
        k_data_off: usize,
        v_data_off: usize,
        q_off: usize,
        out_off: usize,
        cache_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        head_dim_bytes: usize,
        row_bytes: usize,
        scale: f32,
        threads: usize,
    ) -> i32 {
        const BS: usize = 32; // Q8_0_BLOCK_SIZE
        const BB: usize = 34; // Q8_0_BLOCK_BYTES
        if num_kv_heads == 0 || num_q_heads % num_kv_heads != 0 || head_dim % BS != 0 {
            return 0;
        }
        let group = num_q_heads / num_kv_heads;
        let blocks_per_head = head_dim / BS;
        let k_base = base + k_data_off;
        let v_base = base + v_data_off;
        let q_base = base + q_off;
        let out_base = base + out_off;

        // Block helpers — identical to the serial kernel's (NEON on aarch64),
        // so the f32 reduction order is preserved bit-for-bit.
        #[inline]
        unsafe fn dq(src: *const u8, dst: &mut [f32; 32]) {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let scale =
                    half::f16::from_bits(core::ptr::read_unaligned(src as *const u16)).to_f32();
                let sv = vdupq_n_f32(scale);
                let q = src.add(2) as *const i8;
                let d = dst.as_mut_ptr();
                for c in 0..2 {
                    let o = c * 16;
                    let x = vld1q_s8(q.add(o));
                    let lo = vmovl_s8(vget_low_s8(x));
                    let hi = vmovl_s8(vget_high_s8(x));
                    vst1q_f32(
                        d.add(o),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), sv),
                    );
                    vst1q_f32(
                        d.add(o + 4),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), sv),
                    );
                    vst1q_f32(
                        d.add(o + 8),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), sv),
                    );
                    vst1q_f32(
                        d.add(o + 12),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), sv),
                    );
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let scale =
                    half::f16::from_bits(core::ptr::read_unaligned(src as *const u16)).to_f32();
                let q = src.add(2) as *const i8;
                for i in 0..32 {
                    dst[i] = (*q.add(i) as f32) * scale;
                }
            }
        }
        #[inline]
        unsafe fn dotb(q: *const f32, k: &[f32; 32]) -> f32 {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let kp = k.as_ptr();
                let mut acc = vdupq_n_f32(0.0);
                for i in 0..8 {
                    let o = i * 4;
                    acc = vfmaq_f32(acc, vld1q_f32(q.add(o)), vld1q_f32(kp.add(o)));
                }
                vaddvq_f32(acc)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut acc = 0.0f32;
                for i in 0..32 {
                    acc += *q.add(i) * k[i];
                }
                acc
            }
        }
        #[inline]
        unsafe fn axpyb(out: *mut f32, w: f32, v: &[f32; 32]) {
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let wv = vdupq_n_f32(w);
                let vp = v.as_ptr();
                for i in 0..8 {
                    let o = i * 4;
                    vst1q_f32(
                        out.add(o),
                        vfmaq_f32(vld1q_f32(out.add(o)), wv, vld1q_f32(vp.add(o))),
                    );
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..32 {
                    *out.add(i) += w * v[i];
                }
            }
        }

        rayzor_runtime::worker_pool::global().parallel_rows(
            num_kv_heads,
            threads,
            move |lo, hi| {
                let mut scores = std::vec![0.0f32; group * cache_len];
                let mut gmax = std::vec![f32::NEG_INFINITY; group];
                let mut kb = [0.0f32; BS];
                let mut vb = [0.0f32; BS];
                for kv_head in lo..hi {
                    let gstart = kv_head * group;
                    for s in scores.iter_mut() {
                        *s = 0.0;
                    }
                    for m in gmax.iter_mut() {
                        *m = f32::NEG_INFINITY;
                    }
                    // K step: dequant each block once, dot against the group's heads.
                    for l in 0..cache_len {
                        let krow = (k_base + l * row_bytes + kv_head * head_dim_bytes) as *const u8;
                        for b in 0..blocks_per_head {
                            unsafe { dq(krow.add(b * BB), &mut kb) };
                            for gi in 0..group {
                                let qrow = (q_base + (gstart + gi) * head_dim * 4) as *const f32;
                                let p = unsafe { dotb(qrow.add(b * BS), &kb) };
                                scores[gi * cache_len + l] += p;
                            }
                        }
                        for gi in 0..group {
                            let s = scores[gi * cache_len + l] * scale;
                            scores[gi * cache_len + l] = s;
                            if s > gmax[gi] {
                                gmax[gi] = s;
                            }
                        }
                    }
                    // Softmax (libm::expf — matches the guest's fmath::exp on wasm32).
                    for gi in 0..group {
                        let mg = gmax[gi];
                        let row = &mut scores[gi * cache_len..(gi + 1) * cache_len];
                        let mut denom = 0.0f32;
                        for s in row.iter_mut() {
                            let e = libm::expf(*s - mg);
                            *s = e;
                            denom += e;
                        }
                        let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                        for s in row.iter_mut() {
                            *s *= inv;
                        }
                    }
                    // Zero the group's output rows.
                    for gi in 0..group {
                        let orow = (out_base + (gstart + gi) * head_dim * 4) as *mut f32;
                        for d in 0..head_dim {
                            unsafe { *orow.add(d) = 0.0 };
                        }
                    }
                    // V step: dequant each block once, axpy into the group's heads.
                    for l in 0..cache_len {
                        let vrow = (v_base + l * row_bytes + kv_head * head_dim_bytes) as *const u8;
                        for b in 0..blocks_per_head {
                            unsafe { dq(vrow.add(b * BB), &mut vb) };
                            for gi in 0..group {
                                let w = scores[gi * cache_len + l];
                                let orow = (out_base + (gstart + gi) * head_dim * 4) as *mut f32;
                                unsafe { axpyb(orow.add(b * BS), w, &vb) };
                            }
                        }
                    }
                }
            },
        );
        1
    }

    fn looks_like_wasm_ptr(raw: i32) -> bool {
        wasm_addr(raw) > 65536
    }

    fn ret_ptr(ptr: i32, ty: &ValType) -> Val {
        match ty {
            ValType::I64 => Val::I64(wasm_addr(ptr) as i64),
            _ => Val::I32(ptr),
        }
    }

    /// Return an integer in whatever type the WASM import expects.
    fn ret_int(val: i32, ty: &ValType) -> Val {
        match ty {
            ValType::I64 => Val::I64(val as i64),
            ValType::F32 => Val::F32((val as f32).to_bits()),
            ValType::F64 => Val::F64((val as f64).to_bits()),
            _ => Val::I32(val),
        }
    }

    fn ret_f32(val: f32, ty: &ValType) -> Val {
        match ty {
            ValType::F64 => Val::F64((val as f64).to_bits()),
            ValType::I32 => Val::I32(val.to_bits() as i32),
            _ => Val::F32(val.to_bits()),
        }
    }

    fn ret_f64(val: f64, ty: &ValType) -> Val {
        match ty {
            ValType::F32 => Val::F32((val as f32).to_bits()),
            ValType::I64 => Val::I64(val.to_bits() as i64),
            _ => Val::F64(val.to_bits()),
        }
    }

    /// Unbox a DynamicValue pointer from WASM memory.
    /// DynamicValue = { type_id: u32, value_ptr: u32 } at `ptr`.
    /// Types: 0=Void, 1=Null, 2=Bool, 3=Int, 4=Float, 5=String.
    /// Returns the raw i32/i64 value, or the original value if not a pointer.
    /// Read WASM memory at `addr` as a slice of bytes.
    ///
    /// Tries the module's exported `memory` first (normal, non-shared
    /// modules). Falls back to the imported `SharedMemory` we stashed on
    /// `WasmState` — wasmtime does not surface imported shared memory via
    /// `caller.get_export("memory")`.
    fn read_wasm_mem(
        caller: &mut Caller<'_, WasmState>,
        addr: usize,
        len: usize,
    ) -> Option<Vec<u8>> {
        if let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory()) {
            let data = mem.data(&*caller);
            if addr.checked_add(len)? <= data.len() {
                return Some(data[addr..addr + len].to_vec());
            }
            return None;
        }
        let shared = caller.data().shared_memory.clone()?;
        let cells = shared.data();
        if addr.checked_add(len)? > cells.len() {
            return None;
        }
        // Safety: snapshot-read of `len` bytes from linear memory. The
        // runtime schedules worker tasks synchronously on the host call
        // that polls `join` / `is_finished`, so no wasm worker is mutating
        // this region concurrently with the host read.
        let mut out = Vec::with_capacity(len);
        unsafe {
            for i in 0..len {
                out.push(*cells[addr + i].get());
            }
        }
        Some(out)
    }

    /// Unbox a DynamicValue pointer from WASM memory → i32.
    /// DynamicValue = { type_id: u32, value_ptr: u32 }. Types: 2=Bool, 3=Int.
    fn unbox_int_from_memory(caller: &mut Caller<'_, WasmState>, raw: i32) -> i32 {
        // Heuristic: DynamicValue pointers are heap-allocated (> 64KB) and 4-byte aligned.
        // DynamicValue = { type_id: u32 (0-5), value_ptr: u32 }
        let addr = wasm_addr(raw);
        if addr > 65536 && (addr & 3) == 0 {
            if let Some(dv) = read_wasm_mem(caller, addr as usize, 8) {
                let type_id = u32::from_le_bytes(dv[0..4].try_into().unwrap());
                let value_ptr = u32::from_le_bytes(dv[4..8].try_into().unwrap()) as usize;
                // Valid DynamicValue type_ids: 0=Void, 1=Null, 2=Bool, 3=Int, 4=Float, 5=String
                if matches!(type_id, 2 | 3) && value_ptr > 0 {
                    if let Some(vb) = read_wasm_mem(caller, value_ptr, 4) {
                        return i32::from_le_bytes(vb[0..4].try_into().unwrap());
                    }
                }
            }
        }
        raw
    }

    /// Block until a dispatched closure (via `rayzor_thread_spawn`) completes,
    /// caching its result so repeated join/is_finished calls keep working.
    fn join_worker(state: &mut WasmState, h: i32) -> i64 {
        // Clone the Arc and drop the borrow before blocking, so the wait does
        // not hold `state`.
        let slot = match state.thread_handles.get(&h) {
            Some(ts) => {
                if let Some(r) = ts.result {
                    return r;
                }
                ts.slot.clone()
            }
            None => return 0,
        };
        let r = {
            let mut g = slot.done.lock().unwrap();
            while g.is_none() {
                g = slot.cv.wait(g).unwrap();
            }
            g.unwrap()
        };
        if let Some(ts) = state.thread_handles.get_mut(&h) {
            ts.result = Some(r);
        }
        r
    }

    /// Non-blocking: has the dispatched closure finished?
    fn worker_finished(state: &mut WasmState, h: i32) -> i32 {
        match state.thread_handles.get(&h) {
            Some(ts) if ts.result.is_some() => 1,
            Some(ts) => match ts.slot.done.try_lock() {
                Ok(g) if g.is_some() => 1,
                // Unfilled, or momentarily locked while the worker publishes —
                // report "not finished" and let the caller poll again.
                _ => 0,
            },
            None => 0,
        }
    }

    /// Publish a job's result and wake any joiner.
    fn publish(slot: &std::sync::Arc<JobSlot>, r: i64) {
        *slot.done.lock().unwrap() = Some(r);
        slot.cv.notify_all();
    }

    /// One pool worker: instantiate the module once (sharing `shared` memory),
    /// reserve a private stack, then run dispatched closures until the channel
    /// closes. Instantiation and stack reservation happen exactly once — the
    /// whole point of the pool — so per-spawn cost is just an indirect call.
    fn worker_main(
        engine: wasmtime::Engine,
        module: wasmtime::Module,
        shared: wasmtime::SharedMemory,
        rx: std::sync::mpsc::Receiver<Job>,
        worker_id: usize,
    ) {
        let mut store = Store::new(&engine, ());
        let mut linker: Linker<()> = Linker::new(&engine);
        if linker.define(&mut store, "env", "memory", shared).is_err() {
            eprintln!("[wasm-worker {worker_id}] failed to define shared memory");
            return;
        }
        // Workers only run pure-compute closures (e.g. matmul row-bands); stub
        // every other import so an unexpected call traps loudly rather than
        // corrupting shared state.
        let _ = linker.define_unknown_imports_as_traps(&module);
        let inst = match linker.instantiate(&mut store, &module) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("[wasm-worker {worker_id}] instantiate failed: {e}");
                return;
            }
        };
        // Reserve a private 1 MiB stack and point this instance's
        // __stack_pointer at its top (stacks grow down). Reused across all jobs
        // since a worker runs them serially.
        const STACK_SIZE: i32 = 1 << 20;
        if let Some(wasmtime::Extern::Func(f)) = inst.get_export(&mut store, "rayzor_malloc") {
            if let Ok(t) = f.typed::<i32, i32>(&store) {
                if let Ok(base) = t.call(&mut store, STACK_SIZE) {
                    if let Some(wasmtime::Extern::Global(g)) =
                        inst.get_export(&mut store, "__stack_pointer")
                    {
                        let _ = g.set(&mut store, wasmtime::Val::I32(base + STACK_SIZE));
                    }
                }
            }
        }
        let table = match inst.get_export(&mut store, "__indirect_function_table") {
            Some(wasmtime::Extern::Table(t)) => t,
            _ => {
                eprintln!("[wasm-worker {worker_id}] no __indirect_function_table");
                return;
            }
        };
        // Job loop: blocks until a job arrives; exits when all senders drop.
        for job in rx {
            let func = match table.get(&mut store, job.fn_idx as u64) {
                Some(wasmtime::Ref::Func(Some(f))) => f,
                _ => {
                    eprintln!(
                        "[wasm-worker {worker_id}] fn_idx {} not callable",
                        job.fn_idx
                    );
                    publish(&job.slot, 0);
                    continue;
                }
            };
            let r = if let Ok(t) = func.typed::<i32, i64>(&store) {
                t.call(&mut store, job.env_ptr).unwrap_or(0)
            } else if let Ok(t) = func.typed::<i32, i32>(&store) {
                t.call(&mut store, job.env_ptr)
                    .map(|r| r as i64)
                    .unwrap_or(0)
            } else if let Ok(t) = func.typed::<i32, ()>(&store) {
                let _ = t.call(&mut store, job.env_ptr);
                0
            } else {
                0
            };
            publish(&job.slot, r);
        }
    }

    /// Build a pool of `n` worker threads sharing `shared` memory.
    fn build_worker_pool(
        n: usize,
        engine: &wasmtime::Engine,
        module: &wasmtime::Module,
        shared: &wasmtime::SharedMemory,
    ) -> WorkerPool {
        let mut senders = Vec::with_capacity(n);
        for worker_id in 0..n {
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            senders.push(tx);
            let engine = engine.clone();
            let module = module.clone();
            let shared = shared.clone();
            let _ = std::thread::Builder::new()
                .name(format!("rayzor-wasm-worker-{worker_id}"))
                .spawn(move || worker_main(engine, module, shared, rx, worker_id));
        }
        WorkerPool { senders, next: 0 }
    }

    fn unbox_f64_from_memory(caller: &mut Caller<'_, WasmState>, raw: i32) -> f64 {
        let addr = wasm_addr(raw);
        if addr > 65536 && (addr & 3) == 0 {
            if let Some(dv) = read_wasm_mem(caller, addr as usize, 8) {
                let type_id = u32::from_le_bytes(dv[0..4].try_into().unwrap());
                let value_ptr = u32::from_le_bytes(dv[4..8].try_into().unwrap()) as usize;
                if type_id == 4 && value_ptr > 0 {
                    if let Some(vb) = read_wasm_mem(caller, value_ptr, 8) {
                        return f64::from_le_bytes(vb[0..8].try_into().unwrap());
                    }
                }
            }
        }
        raw as f64
    }

    /// Allocate `size` bytes in WASM linear memory. Prefer the merged
    /// runtime-wasm allocator so host-created Haxe objects live in the same
    /// heap as runtime-created objects. Fall back to the runner's top-down
    /// bump region only when the runtime export is unavailable.
    ///
    /// Returns the WASM linear memory address.
    ///
    /// If the would-be allocation underflows past the application heap region
    /// (`host_alloc_ptr < size`), grows the exported `memory` by enough pages
    /// to fit `size` plus a small slack, then re-seats `host_alloc_ptr` at the
    /// new top so the next decrement lands in the freshly-mapped region. The
    /// gap between the previous top and the new top is left unused (small —
    /// bounded by `(size + slack)` rounded up to one page).
    fn host_alloc(caller: &mut Caller<'_, WasmState>, size: u32) -> u32 {
        let size = size.max(1);
        if let Some(ptr) = runtime_malloc(caller, size) {
            return ptr;
        }

        const PAGE: u32 = 65536;
        let mut ptr = caller.data().host_alloc_ptr;
        let low_bound = caller.data().host_alloc_low_bound;
        let would_underflow = ptr < size || (ptr - size) < low_bound;
        if would_underflow {
            // Protect the region allocated so far. After growing, the newly
            // added pages above the old top become the next working region.
            let new_low_bound = ptr;
            let slack: u32 = 16 * 1024 * 1024;
            let need_bytes = size.saturating_add(slack);
            let delta_pages = ((need_bytes as u64 + PAGE as u64 - 1) / PAGE as u64) as u64;
            let mut grew = false;
            if let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                if mem.grow(&mut *caller, delta_pages).is_ok() {
                    let new_size = mem.data_size(&*caller) as u32;
                    caller.data_mut().host_alloc_ptr = new_size - 16;
                    grew = true;
                }
            }
            if !grew {
                if let Some(shared) = caller.data().shared_memory.clone() {
                    if shared.grow(delta_pages).is_ok() {
                        let new_size = shared.data_size() as u32;
                        caller.data_mut().host_alloc_ptr = new_size - 16;
                        grew = true;
                    }
                }
            }
            if grew {
                caller.data_mut().host_alloc_low_bound = new_low_bound;
            }
            ptr = caller.data().host_alloc_ptr;
        }
        let new_ptr = ptr.wrapping_sub(size) & !3;
        caller.data_mut().host_alloc_ptr = new_ptr;
        new_ptr
    }

    fn runtime_malloc(caller: &mut Caller<'_, WasmState>, size: u32) -> Option<u32> {
        let func = caller
            .get_export("rayzor_malloc")
            .and_then(|export| export.into_func())?;
        let malloc = func.typed::<i32, i32>(&*caller).ok()?;
        let ptr = malloc.call(&mut *caller, size as i32).ok()?;
        if ptr > 0 {
            Some(ptr as u32)
        } else {
            None
        }
    }

    /// Write bytes into WASM linear memory at `addr`.
    ///
    /// Prefers the module's exported memory, falling back to the imported
    /// `SharedMemory` stashed on `WasmState` (see `read_wasm_mem` note).
    fn write_wasm_mem(caller: &mut Caller<'_, WasmState>, addr: u32, bytes: &[u8]) {
        if let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory()) {
            let data = mem.data_mut(&mut *caller);
            let a = addr as usize;
            if a + bytes.len() <= data.len() {
                data[a..a + bytes.len()].copy_from_slice(bytes);
            } else {
                eprintln!(
                    "[wasm-runner] write_wasm_mem: out of bounds addr={:#x} len={} data.len={}",
                    addr,
                    bytes.len(),
                    data.len()
                );
            }
            return;
        }
        if let Some(shared) = caller.data().shared_memory.clone() {
            let cells = shared.data();
            let a = addr as usize;
            if a + bytes.len() > cells.len() {
                eprintln!("[wasm-runner] write_wasm_mem: shared out of bounds addr={:#x} len={} cells.len={}",
                    addr, bytes.len(), cells.len());
                return;
            }
            // Safety: mirror of `read_wasm_mem` — host calls are serialized
            // against wasm worker tasks, so no concurrent writer races us.
            unsafe {
                for (i, b) in bytes.iter().enumerate() {
                    *cells[a + i].get() = *b;
                }
            }
            return;
        }
        eprintln!(
            "[wasm-runner] write_wasm_mem: no memory export at addr={:#x}",
            addr
        );
    }

    /// Box an i32 value as a DynamicValue in WASM memory.
    /// Layout: allocate 4 bytes for value, 8 bytes for header {type_id=3, value_ptr}.
    /// Returns the DynamicValue pointer (WASM address).
    fn box_int_in_wasm(caller: &mut Caller<'_, WasmState>, val: i32) -> i32 {
        let val_addr = host_alloc(caller, 4);
        write_wasm_mem(caller, val_addr, &val.to_le_bytes());
        let dv_addr = host_alloc(caller, 8);
        write_wasm_mem(caller, dv_addr, &3u32.to_le_bytes()); // type_id = 3 (Int)
        write_wasm_mem(caller, dv_addr + 4, &val_addr.to_le_bytes()); // value_ptr
                                                                      // Verify the write succeeded
        if let Some(bytes) = read_wasm_mem(caller, dv_addr as usize, 8) {
            eprintln!(
                "[wasm-runner] box_int: dv_addr={:#x} bytes={:?} val_addr={:#x} val={}",
                dv_addr, bytes, val_addr, val
            );
        }
        dv_addr as i32
    }

    /// Box a bool (i32 0/1) as a DynamicValue with type_id=2 (Bool).
    fn box_bool_in_wasm(caller: &mut Caller<'_, WasmState>, val: i32) -> i32 {
        let val_addr = host_alloc(caller, 4);
        write_wasm_mem(caller, val_addr, &val.to_le_bytes());
        let dv_addr = host_alloc(caller, 8);
        write_wasm_mem(caller, dv_addr, &2u32.to_le_bytes()); // type_id = 2 (Bool)
        write_wasm_mem(caller, dv_addr + 4, &val_addr.to_le_bytes());
        dv_addr as i32
    }

    /// Box an f64 value as a DynamicValue in WASM memory.
    fn box_float_in_wasm(caller: &mut Caller<'_, WasmState>, val: f64) -> i32 {
        let val_addr = host_alloc(caller, 8);
        write_wasm_mem(caller, val_addr, &val.to_le_bytes());
        let dv_addr = host_alloc(caller, 8);
        write_wasm_mem(caller, dv_addr, &4u32.to_le_bytes()); // type_id = 4 (Float)
        write_wasm_mem(caller, dv_addr + 4, &val_addr.to_le_bytes()); // value_ptr
        dv_addr as i32
    }

    fn box_dynamic_header_in_wasm(
        caller: &mut Caller<'_, WasmState>,
        type_id: u32,
        value_ptr: u32,
    ) -> i32 {
        let dv_addr = host_alloc(caller, 8);
        write_wasm_mem(caller, dv_addr, &type_id.to_le_bytes());
        write_wasm_mem(caller, dv_addr + 4, &value_ptr.to_le_bytes());
        dv_addr as i32
    }

    fn box_int_quiet_in_wasm(caller: &mut Caller<'_, WasmState>, val: i32) -> i32 {
        let val_addr = host_alloc(caller, 4);
        write_wasm_mem(caller, val_addr, &val.to_le_bytes());
        box_dynamic_header_in_wasm(caller, 3, val_addr)
    }

    fn box_bool_quiet_in_wasm(caller: &mut Caller<'_, WasmState>, val: bool) -> i32 {
        let val_addr = host_alloc(caller, 4);
        let raw = if val { 1i32 } else { 0i32 };
        write_wasm_mem(caller, val_addr, &raw.to_le_bytes());
        box_dynamic_header_in_wasm(caller, 2, val_addr)
    }

    fn read_haxe_string_raw(caller: &mut Caller<'_, WasmState>, ptr: i32) -> Option<String> {
        let addr = wasm_addr(ptr);
        if addr == 0 {
            return None;
        }
        let header = read_wasm_mem(caller, addr as usize, 8)?;
        let data_ptr = u32::from_le_bytes(header[0..4].try_into().ok()?) as usize;
        let len = u32::from_le_bytes(header[4..8].try_into().ok()?) as usize;
        if len == 0 {
            return Some(String::new());
        }
        if len > 64 * 1024 * 1024 || data_ptr == 0 {
            return None;
        }
        let bytes = read_wasm_mem(caller, data_ptr, len)?;
        Some(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Read a HaxeString { data_ptr: u32, len: u32 } from WASM memory → Rust String.
    fn read_haxe_string(caller: &mut Caller<'_, WasmState>, str_ptr: i32) -> String {
        let ptr = unbox_int_from_memory(caller, str_ptr);
        read_haxe_string_raw(caller, ptr).unwrap_or_default()
    }

    fn format_std_string_ptr(caller: &mut Caller<'_, WasmState>, raw: i32) -> String {
        let addr = wasm_addr(raw);
        if addr > 65536 && (addr & 3) == 0 {
            if let Some(dv) = read_wasm_mem(caller, addr as usize, 8) {
                let type_id = u32::from_le_bytes(dv[0..4].try_into().unwrap());
                let value_ptr = u32::from_le_bytes(dv[4..8].try_into().unwrap()) as i32;
                match type_id {
                    1 => return "null".to_string(),
                    2 => return (unbox_int_from_memory(caller, raw) != 0).to_string(),
                    3 => return unbox_int_from_memory(caller, raw).to_string(),
                    4 => {
                        if value_ptr > 0 {
                            if let Some(bytes) = read_wasm_mem(caller, value_ptr as usize, 8) {
                                let mut arr = [0u8; 8];
                                arr.copy_from_slice(&bytes);
                                return f64::from_le_bytes(arr).to_string();
                            }
                        }
                    }
                    5 => {
                        if let Some(s) = read_haxe_string_raw(caller, value_ptr) {
                            return s;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(s) = read_haxe_string_raw(caller, raw) {
                return s;
            }
        }
        raw.to_string()
    }

    /// Write a Rust string into WASM memory as HaxeString { data_ptr, len, cap }.
    /// Returns the HaxeString struct pointer.
    fn write_haxe_string(caller: &mut Caller<'_, WasmState>, s: &str) -> i32 {
        let bytes = s.as_bytes();
        let data_addr = host_alloc(caller, bytes.len() as u32);
        write_wasm_mem(caller, data_addr, bytes);
        let struct_addr = host_alloc(caller, 12);
        write_wasm_mem(caller, struct_addr, &data_addr.to_le_bytes());
        write_wasm_mem(caller, struct_addr + 4, &(bytes.len() as u32).to_le_bytes());
        write_wasm_mem(caller, struct_addr + 8, &(bytes.len() as u32).to_le_bytes());
        struct_addr as i32
    }

    /// Write the canonical 40-byte `HaxeBytes` header at `addr` so that the
    /// runtime-wasm cdylib (`runtime-wasm/src/qtensor.rs`, `tensor.rs`) can
    /// dereference it as `*const HaxeBytes` and find `{ptr, len, cap}` at the
    /// expected offsets. Trailing fields (kind/owner/refcount/fd) are set to
    /// a benign default; runtime-wasm only reads `ptr` and `len` in the
    /// no-copy code paths we care about.
    fn write_haxe_bytes_header(
        caller: &mut Caller<'_, WasmState>,
        header_addr: u32,
        data_addr: u32,
        len: u32,
        cap: u32,
    ) {
        write_wasm_mem(caller, header_addr, &data_addr.to_le_bytes());
        write_wasm_mem(caller, header_addr + 4, &len.to_le_bytes());
        write_wasm_mem(caller, header_addr + 8, &cap.to_le_bytes());
        write_wasm_mem(caller, header_addr + 12, &[0u8; 8]);
        write_wasm_mem(caller, header_addr + 20, &0u32.to_le_bytes());
        write_wasm_mem(caller, header_addr + 24, &1i64.to_le_bytes());
        write_wasm_mem(caller, header_addr + 32, &(-1i32).to_le_bytes());
        write_wasm_mem(caller, header_addr + 36, &0u32.to_le_bytes());
    }

    /// Allocate a HaxeBytes header + owned data block in wasm linear memory.
    /// Returns the header pointer (used as the "handle" by Haxe code).
    fn write_haxe_bytes_in_wasm(caller: &mut Caller<'_, WasmState>, data: &[u8]) -> i32 {
        let data_addr = host_alloc(caller, data.len().max(1) as u32);
        if !data.is_empty() {
            write_wasm_mem(caller, data_addr, data);
        }
        let header_addr = host_alloc(caller, 40);
        write_haxe_bytes_header(
            caller,
            header_addr,
            data_addr,
            data.len() as u32,
            data.len() as u32,
        );
        header_addr as i32
    }

    /// Allocate a HaxeBytes header that shares data with an existing buffer
    /// (no copy). Used for `sub` / `subWithBase` so a 200MB GGUF doesn't
    /// duplicate per tensor.
    fn make_haxe_bytes_view(caller: &mut Caller<'_, WasmState>, data_addr: u32, len: u32) -> i32 {
        let header_addr = host_alloc(caller, 40);
        write_haxe_bytes_header(caller, header_addr, data_addr, len, len);
        header_addr as i32
    }

    /// Read `{data_ptr, len, cap}` from a HaxeBytes header. Returns None on
    /// bad reads (e.g. address out of memory range).
    fn read_haxe_bytes_header(
        caller: &mut Caller<'_, WasmState>,
        header_ptr: i32,
    ) -> Option<(u32, u32, u32)> {
        let addr = wasm_addr(header_ptr);
        if addr == 0 {
            return None;
        }
        let hdr = read_wasm_mem(caller, addr as usize, 12)?;
        let data_ptr = u32::from_le_bytes(hdr[0..4].try_into().ok()?);
        let len = u32::from_le_bytes(hdr[4..8].try_into().ok()?);
        let cap = u32::from_le_bytes(hdr[8..12].try_into().ok()?);
        Some((data_ptr, len, cap))
    }

    /// Dual-mode read of a byte slice from a bytes handle:
    /// - If handle is a wasm pointer (> 65536), read via HaxeBytes header.
    /// - Otherwise, fall back to the legacy host-side `bytes_handles` table.
    fn read_bytes_slice(
        caller: &mut Caller<'_, WasmState>,
        handle: i32,
        pos: usize,
        len: usize,
    ) -> Vec<u8> {
        if looks_like_wasm_ptr(handle) {
            if let Some((data_addr, total_len, _)) = read_haxe_bytes_header(caller, handle) {
                let end = pos.saturating_add(len).min(total_len as usize);
                if pos < end {
                    if let Some(v) = read_wasm_mem(caller, data_addr as usize + pos, end - pos) {
                        let mut out = v;
                        out.resize(len, 0);
                        return out;
                    }
                }
            }
            return vec![0u8; len];
        }
        caller
            .data()
            .bytes_handles
            .get(&handle)
            .map(|v| {
                let end = pos.saturating_add(len).min(v.len());
                if pos < end {
                    let mut out = v[pos..end].to_vec();
                    out.resize(len, 0);
                    out
                } else {
                    vec![0u8; len]
                }
            })
            .unwrap_or_else(|| vec![0u8; len])
    }

    /// Dual-mode write of a byte slice into a bytes handle.
    /// Silently truncates if `pos + data.len()` exceeds the bytes' length.
    fn write_bytes_slice(caller: &mut Caller<'_, WasmState>, handle: i32, pos: usize, data: &[u8]) {
        if looks_like_wasm_ptr(handle) {
            if let Some((data_addr, total_len, _)) = read_haxe_bytes_header(caller, handle) {
                let end = pos.saturating_add(data.len()).min(total_len as usize);
                if pos < end {
                    write_wasm_mem(caller, data_addr + pos as u32, &data[..end - pos]);
                }
            }
            return;
        }
        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&handle) {
            let end = pos.saturating_add(data.len()).min(v.len());
            if pos < end {
                v[pos..end].copy_from_slice(&data[..end - pos]);
            }
        }
    }

    /// Dual-mode fill of a byte range with `val`.
    fn fill_bytes_slice(
        caller: &mut Caller<'_, WasmState>,
        handle: i32,
        pos: usize,
        len: usize,
        val: u8,
    ) {
        if looks_like_wasm_ptr(handle) {
            if let Some((data_addr, total_len, _)) = read_haxe_bytes_header(caller, handle) {
                let end = pos.saturating_add(len).min(total_len as usize);
                if pos < end {
                    let buf = vec![val; end - pos];
                    write_wasm_mem(caller, data_addr + pos as u32, &buf);
                }
            }
            return;
        }
        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&handle) {
            let end = pos.saturating_add(len).min(v.len());
            if pos < end {
                v[pos..end].fill(val);
            }
        }
    }

    /// Dual-mode total length of a bytes handle.
    fn bytes_total_len(caller: &mut Caller<'_, WasmState>, handle: i32) -> usize {
        if looks_like_wasm_ptr(handle) {
            return read_haxe_bytes_header(caller, handle)
                .map(|(_, len, _)| len as usize)
                .unwrap_or(0);
        }
        caller
            .data()
            .bytes_handles
            .get(&handle)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Read a HaxeArray of i32 values from WASM memory.
    /// HaxeArray layout: { data_ptr: u32, len: u32, cap: u32, elem_size: u32 }.
    /// Read a raw contiguous block of `len` integer values from WASM memory.
    /// Used for the decomposed `(data_ptr, len)` ABI emitted by the tensor MIR
    /// wrappers, which extract data and length from a HaxeArray *before* the
    /// extern call. Each element is i64-strided to match the native runtime
    /// (`runtime/src/tensor.rs::read_shape`); on 32-bit WASM the high 4 bytes
    /// are zero, so we read the low 4 bytes per slot.
    fn read_raw_int_array_i64stride(
        caller: &mut Caller<'_, WasmState>,
        data_ptr: i32,
        len: i32,
    ) -> Vec<i32> {
        let addr = wasm_addr(data_ptr);
        if addr == 0 || len <= 0 {
            return vec![];
        }
        let stride = 8usize;
        let total = len as usize * stride;
        if let Some(data) = read_wasm_mem(caller, addr as usize, total) {
            return (0..len as usize)
                .map(|i| {
                    i32::from_le_bytes(
                        data[i * stride..i * stride + 4]
                            .try_into()
                            .unwrap_or([0; 4]),
                    )
                })
                .collect();
        }
        vec![]
    }

    /// Read a raw contiguous block of `len` f64 values from WASM memory
    /// (i64-strided slots, low 8 bytes of each slot is the f64 payload).
    fn read_raw_f64_array_i64stride(
        caller: &mut Caller<'_, WasmState>,
        data_ptr: i32,
        len: i32,
    ) -> Vec<f64> {
        let addr = wasm_addr(data_ptr);
        if addr == 0 || len <= 0 {
            return vec![];
        }
        let stride = 8usize;
        let total = len as usize * stride;
        if let Some(data) = read_wasm_mem(caller, addr as usize, total) {
            return (0..len as usize)
                .map(|i| {
                    f64::from_le_bytes(
                        data[i * stride..i * stride + 8]
                            .try_into()
                            .unwrap_or([0; 8]),
                    )
                })
                .collect();
        }
        vec![]
    }

    fn read_haxe_array_i32(caller: &mut Caller<'_, WasmState>, arr_ptr: i32) -> Vec<i32> {
        let ptr = wasm_addr_usize(unbox_int_from_memory(caller, arr_ptr));
        if ptr == 0 {
            return vec![];
        }
        // HaxeArray layout (32 bytes, MIR i64 stride):
        // ptr at offset 0, len at offset 8, cap at offset 16, elem_size at offset 24
        if let Some(header) = read_wasm_mem(caller, ptr, 32) {
            let data_ptr = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let len = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
            let elem_size = u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize;
            let actual_size = if elem_size > 0 { elem_size } else { 8 };
            if let Some(data) = read_wasm_mem(caller, data_ptr, len * actual_size) {
                return (0..len)
                    .map(|i| {
                        i32::from_le_bytes(
                            data[i * actual_size..i * actual_size + 4]
                                .try_into()
                                .unwrap_or([0; 4]),
                        )
                    })
                    .collect();
            }
        }
        vec![]
    }

    /// Read a HaxeArray of f64 values from WASM memory.
    /// HaxeArray layout (32 bytes, MIR i64 stride):
    /// ptr at offset 0, len at offset 8, cap at offset 16, elem_size at offset 24
    fn read_haxe_array_f64(caller: &mut Caller<'_, WasmState>, arr_ptr: i32) -> Vec<f64> {
        let ptr = wasm_addr_usize(unbox_int_from_memory(caller, arr_ptr));
        if ptr == 0 {
            return vec![];
        }
        let (data_ptr, len, elem_size) = if let Some(header) = read_wasm_mem(caller, ptr, 32) {
            (
                u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize,
                u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize,
                u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize,
            )
        } else {
            return vec![];
        };
        let actual_size = if elem_size > 0 { elem_size } else { 8 };
        if let Some(data) = read_wasm_mem(caller, data_ptr, len * actual_size) {
            // Elements stored as f64 directly (from haxe_array_push_f64) for 8-byte slots
            return (0..len)
                .map(|i| {
                    if actual_size >= 8 {
                        f64::from_le_bytes(
                            data[i * actual_size..i * actual_size + 8]
                                .try_into()
                                .unwrap_or([0; 8]),
                        )
                    } else {
                        f32::from_le_bytes(
                            data[i * actual_size..i * actual_size + 4]
                                .try_into()
                                .unwrap_or([0; 4]),
                        ) as f64
                    }
                })
                .collect();
        }
        vec![]
    }

    // -- Engine & module setup --
    let mut config = Config::new();
    // Force the Cranelift JIT. wasmtime's default Strategy::Auto can pick
    // the Pulley bytecode interpreter (~50x slower) when both are compiled
    // in; we always want the optimizing JIT for the heavy quant matmuls.
    config.strategy(wasmtime::Strategy::Cranelift);
    config.cranelift_opt_level(wasmtime::OptLevel::Speed);
    if std::env::var_os("RAYZOR_WASM_ENGINE_DEBUG").is_some() {
        eprintln!("[wasm-runner] strategy=Cranelift opt=Speed");
    }
    config.wasm_simd(true);
    // Enable threads + bulk-memory + shared-memory for the shared-memory
    // runtime build. Without these wasmtime rejects the module during
    // compilation because the memory section carries the `shared` flag and
    // the function bodies use `memory.init` / `data.drop` against passive
    // data segments.
    config.wasm_threads(true);
    config.wasm_bulk_memory(true);
    config.shared_memory(true);
    // Reserve the full 32-bit address space (+ guard) so linear-memory
    // accesses are bounds-checked by the guard page rather than an
    // explicit compare+branch on every load. The quant matmul is
    // memory-bound (billions of loads/token); without elision those
    // per-load checks dominate and the engine choice (JIT vs interpreter)
    // becomes irrelevant. 4 GiB reservation + 32 MiB guard, memory pinned.
    config.memory_reservation(1 << 32); // 4 GiB
    config.memory_guard_size(1 << 25); // 32 MiB
    config.memory_may_move(false);
    let engine = Engine::new(&config).map_err(|e| format!("Engine config failed: {}", e))?;
    let _compile_t0 = std::time::Instant::now();
    let module =
        Module::new(&engine, wasm_bytes).map_err(|e| format!("WASM compilation failed: {}", e))?;
    if std::env::var_os("RAYZOR_WASM_ENGINE_DEBUG").is_some() {
        eprintln!(
            "[wasm-runner] module compiled in {:.3}s ({} KB) — long compile = Cranelift JIT, instant = interpreter",
            _compile_t0.elapsed().as_secs_f64(),
            wasm_bytes.len() / 1024
        );
    }

    // -- WASI context --
    let mut builder = wasi_common::WasiCtxBuilder::new();
    builder.inherit_stdio().inherit_env();
    if let Ok(cwd) = std::env::current_dir() {
        let _ = builder.preopened_dir(
            &cwd,
            ".",
            wasi_common::DirPerms::all(),
            wasi_common::FilePerms::all(),
        );
    }

    let state = WasmState {
        wasi_ctx: builder.build_p1(),
        bytes_handles: BTreeMap::new(),
        next_bytes_id: 1,
        ereg_handles: BTreeMap::new(),
        next_ereg_id: 1,
        mutex_handles: BTreeMap::new(),
        next_mutex_id: 1,
        tensor_handles: BTreeMap::new(),
        next_tensor_id: 1,
        thread_handles: BTreeMap::new(),
        next_thread_id: 1,
        worker_pool: None,
        wgpu_ctx: {
            // wgpu/Metal init spawns driver threads and allocates GPU resources
            // up front; the quant matmul path runs entirely on the CPU host
            // worker pool, so RAYZOR_WASM_NO_WGPU=1 skips it (faster startup,
            // one fewer source of host-side nondeterminism).
            if std::env::var("RAYZOR_WASM_NO_WGPU").as_deref() == Ok("1") {
                None
            } else {
                let ctx = WgpuComputeCtx::new();
                if ctx.is_some() {
                    eprintln!("[wasm-runner] wgpu compute backend initialized (Metal/Vulkan/DX12)");
                }
                ctx
            }
        },
        host_alloc_ptr: 0,
        host_alloc_low_bound: 4 * 1024 * 1024, // reserve bottom 4 MiB for app heap
        shared_memory: None,
        program_args: program_args.to_vec(),
        stringmap_handles: BTreeMap::new(),
        next_stringmap_id: 1,
        side_instance: None,
        engine: engine.clone(),
        module: module.clone(),
    };
    let mut store = Store::new(&engine, state);

    // -- Linker: WASI P1 --
    let mut linker = Linker::new(&engine);
    wasi_common::p1::add_to_linker_sync(&mut linker, |s: &mut WasmState| &mut s.wasi_ctx)
        .map_err(|e| format!("WASI linker error: {}", e))?;

    // -- Collect rayzor imports --
    let rayzor_imports: Vec<(String, FuncType)> = module
        .imports()
        .filter(|i| i.module() == "rayzor")
        .filter_map(|i| match i.ty() {
            ExternType::Func(ft) => Some((i.name().to_string(), ft)),
            _ => None,
        })
        .collect();
    let rayzor_import_names: BTreeSet<String> = rayzor_imports
        .iter()
        .map(|(name, _)| name.clone())
        .collect();

    // -- Register Bytes host functions --
    // Map bare names to their canonical qualified names
    fn canonical_bytes_name(name: &str) -> Option<&str> {
        match name {
            // Qualified names (snake_case — canonical form from WASM backend)
            "haxe_bytes_alloc" => Some("haxe_bytes_alloc"),
            "haxe_bytes_length" => Some("haxe_bytes_length"),
            "haxe_bytes_of_string" => Some("haxe_bytes_of_string"),
            "haxe_bytes_get" => Some("haxe_bytes_get"),
            "haxe_bytes_set" => Some("haxe_bytes_set"),
            "haxe_bytes_get_int16" => Some("haxe_bytes_get_int16"),
            "haxe_bytes_set_int16" => Some("haxe_bytes_set_int16"),
            "haxe_bytes_get_int32" => Some("haxe_bytes_get_int32"),
            "haxe_bytes_set_int32" => Some("haxe_bytes_set_int32"),
            "haxe_bytes_get_int64" => Some("haxe_bytes_get_int64"),
            "haxe_bytes_set_int64" => Some("haxe_bytes_set_int64"),
            "haxe_bytes_get_float" => Some("haxe_bytes_get_float"),
            "haxe_bytes_set_float" => Some("haxe_bytes_set_float"),
            "haxe_bytes_get_double" => Some("haxe_bytes_get_double"),
            "haxe_bytes_set_double" => Some("haxe_bytes_set_double"),
            "haxe_bytes_fill" => Some("haxe_bytes_fill"),
            "haxe_bytes_blit" => Some("haxe_bytes_blit"),
            "haxe_bytes_compare" => Some("haxe_bytes_compare"),
            "haxe_bytes_sub" => Some("haxe_bytes_sub"),
            "haxe_bytes_sub_base_u64lh" => Some("haxe_bytes_sub_base_u64lh"),
            "haxe_bytes_to_string" => Some("haxe_bytes_to_string"),
            // Bare names (from runtime-wasm module imports surviving linker merge)
            "alloc" => Some("haxe_bytes_alloc"),
            "ofString" => Some("haxe_bytes_of_string"),
            "length" => Some("haxe_bytes_length"),
            "get" => Some("haxe_bytes_get"),
            "set" => Some("haxe_bytes_set"),
            "getInt16" => Some("haxe_bytes_get_int16"),
            "setInt16" => Some("haxe_bytes_set_int16"),
            "getInt32" => Some("haxe_bytes_get_int32"),
            "setInt32" => Some("haxe_bytes_set_int32"),
            "getInt64" => Some("haxe_bytes_get_int64"),
            "setInt64" => Some("haxe_bytes_set_int64"),
            "getFloat" => Some("haxe_bytes_get_float"),
            "setFloat" => Some("haxe_bytes_set_float"),
            "getDouble" => Some("haxe_bytes_get_double"),
            "setDouble" => Some("haxe_bytes_set_double"),
            "fill" => Some("haxe_bytes_fill"),
            "blit" => Some("haxe_bytes_blit"),
            "compare" => Some("haxe_bytes_compare"),
            "sub" => Some("haxe_bytes_sub"),
            _ => None,
        }
    }

    let mut registered: BTreeSet<String> = BTreeSet::new();

    // -- Register host-side parallel quant matmul --
    // runtime-wasm's threaded matmul hands us raw wasm byte-offsets and we run
    // the embarrassingly-parallel output-column loop on the NATIVE worker pool
    // (native kernels, real OS threads) over the shared wasm memory. This is
    // the wasm "threading" lever: the single guest thread blocks in this call
    // while host threads do the matmul. Registered here (and marked done so the
    // later stub loop won't redefine it) rather than in the tensor-fallback
    // loop, which is skipped wholesale when runtime-wasm provides the tensors.
    if rayzor_import_names.contains("rayzor_host_qmatmul_par") {
        let par_ty = FuncType::new(
            &engine,
            std::iter::repeat_n(ValType::I32, 9),
            std::iter::once(ValType::I32),
        );
        linker
            .func_new(
                "rayzor",
                "rayzor_host_qmatmul_par",
                par_ty,
                move |mut caller, params, results| {
                    // ADDRESS params are wasm linear-memory offsets: UNSIGNED
                    // u32 spanning the full 0..4GiB range. Zero-extend (`as u32
                    // as usize`) — NOT `.max(0)`. `.max(0)` clamps any offset
                    // with the high bit set (a tensor allocated above 2GiB) to
                    // 0, which would make the host fan its f32 results out to
                    // `base + 0..` and smash the low static data segment. Latent
                    // today (the 1B model peaks under 2GiB so no tensor is
                    // allocated that high), but a real defect for larger models
                    // or contexts that push the wasm heap break past 2GiB.
                    let x_data_off = val_i32(&params[0]) as u32 as usize;
                    let x_stride0 = val_i32(&params[1]).max(0) as usize;
                    let batch = val_i32(&params[2]).max(0) as usize;
                    let k = val_i32(&params[3]).max(0) as usize;
                    let qt_data_off = val_i32(&params[4]) as u32 as usize;
                    let scheme = val_i32(&params[5]);
                    let n = val_i32(&params[6]).max(0) as usize;
                    let y_data_off = val_i32(&params[7]) as u32 as usize;
                    let threads = val_i32(&params[8]);
                    // Diagnostic: RAYZOR_WASM_NO_HOST_MATMUL=1 forces the guest
                    // to fall back to its sequential in-wasm kernel (isolates
                    // host-path correctness/perf from the guest path).
                    if std::env::var("RAYZOR_WASM_NO_HOST_MATMUL").as_deref() == Ok("1") {
                        results[0] = Val::I32(0);
                        return Ok(());
                    }
                    // Stable mmap base of the shared linear memory.
                    let base = match caller.data().shared_memory.clone() {
                        Some(shared) => shared.data().as_ptr() as *const u8 as usize,
                        None => {
                            results[0] = Val::I32(0);
                            return Ok(());
                        }
                    };
                    let handled = unsafe {
                        host_parallel_qmatmul(
                            base,
                            x_data_off,
                            x_stride0,
                            batch,
                            k,
                            qt_data_off,
                            scheme,
                            n,
                            y_data_off,
                            threads,
                        )
                    };
                    results[0] = Val::I32(handled);
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register rayzor_host_qmatmul_par: {}", e))?;
        registered.insert("rayzor_host_qmatmul_par".to_string());
    }

    // -- Register host-side parallel Q8 flash attention --
    // 12 i32 params: k_off, v_off, q_off, out_off, cache_len, num_q_heads,
    // num_kv_heads, head_dim, head_dim_bytes, row_bytes, scale_bits (f32 bits),
    // threads -> i32 (1 handled, 0 fall back to the guest serial kernel).
    if rayzor_import_names.contains("rayzor_host_flash_attn_q8_par") {
        let fty = FuncType::new(
            &engine,
            std::iter::repeat_n(ValType::I32, 12),
            std::iter::once(ValType::I32),
        );
        linker
            .func_new(
                "rayzor",
                "rayzor_host_flash_attn_q8_par",
                fty,
                move |mut caller, params, results| {
                    // Two opt-outs both fall back to the serial guest kernel:
                    // NO_HOST_MATMUL (global host-offload off) and NO_HOST_FLASH
                    // (isolate the attention A/B while keeping host matmul on).
                    if std::env::var("RAYZOR_WASM_NO_HOST_MATMUL").as_deref() == Ok("1")
                        || std::env::var("RAYZOR_WASM_NO_HOST_FLASH").as_deref() == Ok("1")
                    {
                        results[0] = Val::I32(0);
                        return Ok(());
                    }
                    let k_off = val_i32(&params[0]) as u32 as usize;
                    let v_off = val_i32(&params[1]) as u32 as usize;
                    let q_off = val_i32(&params[2]) as u32 as usize;
                    let out_off = val_i32(&params[3]) as u32 as usize;
                    let cache_len = val_i32(&params[4]).max(0) as usize;
                    let num_q_heads = val_i32(&params[5]).max(0) as usize;
                    let num_kv_heads = val_i32(&params[6]).max(0) as usize;
                    let head_dim = val_i32(&params[7]).max(0) as usize;
                    let head_dim_bytes = val_i32(&params[8]).max(0) as usize;
                    let row_bytes = val_i32(&params[9]).max(0) as usize;
                    let scale = f32::from_bits(val_i32(&params[10]) as u32);
                    let threads = val_i32(&params[11]);
                    let nthreads = if threads > 1 {
                        (threads as usize).min(64)
                    } else {
                        rayzor_runtime::worker_pool::auto_kernel_threads()
                    };
                    let base = match caller.data().shared_memory.clone() {
                        Some(shared) => shared.data().as_ptr() as *const u8 as usize,
                        None => {
                            results[0] = Val::I32(0);
                            return Ok(());
                        }
                    };
                    let handled = unsafe {
                        host_parallel_flash_q8(
                            base,
                            k_off,
                            v_off,
                            q_off,
                            out_off,
                            cache_len,
                            num_q_heads,
                            num_kv_heads,
                            head_dim,
                            head_dim_bytes,
                            row_bytes,
                            scale,
                            nthreads,
                        )
                    };
                    results[0] = Val::I32(handled);
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register rayzor_host_flash_attn_q8_par: {}", e))?;
        registered.insert("rayzor_host_flash_attn_q8_par".to_string());
    }

    // -- Q8 KV-cache capability trampolines (dylink.0 side-module) --
    // These guest imports are EXPORTED by the nue-plugins PIC side-module that
    // the side-module load block (after main instantiation) links into this
    // Store. Registering real trampolines here — and marking them `registered`
    // — bypasses the return-0 fallback stub loop at the end, so the guest's Q8
    // calls dispatch into the side instance instead of silently returning null.
    // If no side-module is loaded at run time (capability absent), the
    // trampoline returns 0 and the Haxe side falls back to the F32 KV cache.
    // NOTE: rayzor_tensor_flash_attn_decode_q8 is deliberately ABSENT — the main
    // runtime-wasm module now exports a PARALLEL in-guest implementation
    // (qtensor.rs, bands kv_heads across the worker pool), which the linker
    // resolves ahead of the side-module's serial copy. Keep the cache-lifecycle
    // fns trampolined to the side module (it owns the Q8 buffer).
    const Q8_SIDE_FNS: &[&str] = &[
        "rayzor_kv_cache_q8_alloc",
        "rayzor_kv_cache_q8_append",
        "rayzor_kv_cache_q8_dequant_view",
        "rayzor_kv_cache_q8_free",
    ];
    for fname in Q8_SIDE_FNS {
        let func_ty = match rayzor_imports.iter().find(|(n, _)| n == fname) {
            Some((_, ft)) => ft.clone(),
            None => continue, // guest doesn't reference this capability
        };
        let ret_types: Vec<ValType> = func_ty.results().collect();
        let name_owned = fname.to_string();
        linker
            .func_new(
                "rayzor",
                fname,
                func_ty,
                move |mut caller, params, results| {
                    // Copy the side Instance out of WasmState (16-byte handle)
                    // to drop the &WasmState borrow before taking &mut caller.
                    let side = match caller.data().side_instance {
                        Some(s) => s,
                        None => {
                            for (r, t) in results.iter_mut().zip(&ret_types) {
                                *r = match t {
                                    ValType::I64 => Val::I64(0),
                                    ValType::F32 => Val::F32(0),
                                    ValType::F64 => Val::F64(0),
                                    _ => Val::I32(0),
                                };
                            }
                            return Ok(());
                        }
                    };
                    let f = side.get_func(&mut caller, &name_owned).ok_or_else(|| {
                        wasmtime::Error::msg(format!("side export {name_owned} missing"))
                    })?;
                    // Marshal across the ABI boundary: the guest passes Haxe Int
                    // (i32) but the side export takes i64; coerce each arg to the
                    // side's param type and each result back to the guest's.
                    let side_ty = f.ty(&caller);
                    let side_params: Vec<ValType> = side_ty.params().collect();
                    let side_results: Vec<ValType> = side_ty.results().collect();
                    drop(side_ty);
                    let args: Vec<Val> = params
                        .iter()
                        .zip(&side_params)
                        .map(|(p, t)| coerce_val(p, t))
                        .collect();
                    let mut out: Vec<Val> = side_results
                        .iter()
                        .map(|t| match t {
                            ValType::I64 => Val::I64(0),
                            ValType::F32 => Val::F32(0),
                            ValType::F64 => Val::F64(0),
                            _ => Val::I32(0),
                        })
                        .collect();
                    f.call(&mut caller, &args, &mut out).map_err(|e| {
                        wasmtime::Error::msg(format!("side {name_owned} call: {e}"))
                    })?;
                    if std::env::var_os("RAYZOR_Q8_TRACE").is_some() {
                        eprintln!("[q8-tramp] {name_owned} args={args:?} -> {out:?}");
                    }
                    for (r, (o, gt)) in results.iter_mut().zip(out.iter().zip(&ret_types)) {
                        *r = coerce_val(o, gt);
                    }
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register Q8 trampoline {fname}: {e}"))?;
        registered.insert(fname.to_string());
    }

    for (name, func_ty) in &rayzor_imports {
        let canon = match canonical_bytes_name(name) {
            Some(c) => c,
            None => continue,
        };

        let ret_ty: ValType = func_ty.results().next().unwrap_or(ValType::I32);

        match canon {
            // -- alloc(size) -> handle --
            "haxe_bytes_alloc" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let size = val_i32(&params[0]).max(0) as usize;
                            let header_ptr =
                                write_haxe_bytes_in_wasm(&mut caller, &vec![0u8; size]);
                            results[0] = ret_ptr(header_ptr, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- length(handle) -> i32 --
            "haxe_bytes_length" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let len = bytes_total_len(&mut caller, h) as i32;
                            results[0] = ret_int(len, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- ofString(str_ptr) -> handle --
            "haxe_bytes_of_string" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let str_ptr = wasm_addr_usize(val_i32(&params[0]));
                            // Read HaxeString { data_ptr: i32, len: i32, cap: i32 } via
                            // the read_wasm_mem helper so we transparently pick up
                            // either exported or imported-shared linear memory.
                            let bytes = if let Some(header) = read_wasm_mem(&mut caller, str_ptr, 8)
                            {
                                let data_ptr =
                                    u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
                                let len =
                                    u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
                                read_wasm_mem(&mut caller, data_ptr, len).unwrap_or_default()
                            } else {
                                vec![]
                            };
                            let header_ptr = write_haxe_bytes_in_wasm(&mut caller, &bytes);
                            results[0] = ret_ptr(header_ptr, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- get(handle, pos) -> byte --
            // Bytes methods receive raw primitives from MIR (the builtin mapping
            // carries proper primitive types) and return raw primitives. No boxing
            // at the host boundary — MIR sees values directly.
            "haxe_bytes_get" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let buf = read_bytes_slice(&mut caller, h, pos, 1);
                            results[0] = ret_int(buf[0] as i32, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- set(handle, pos, val) --
            "haxe_bytes_set" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let val = val_i32(&params[2]) as u8;
                            write_bytes_slice(&mut caller, h, pos, &[val]);
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getInt16(handle, pos) -> i32 --
            "haxe_bytes_get_int16" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let buf = read_bytes_slice(&mut caller, h, pos, 2);
                            let val = i16::from_le_bytes(buf[0..2].try_into().unwrap()) as i32;
                            results[0] = ret_int(val, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt16(handle, pos, val) --
            "haxe_bytes_set_int16" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let val = val_i32(&params[2]) as i16;
                            write_bytes_slice(&mut caller, h, pos, &val.to_le_bytes());
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getInt32(handle, pos) -> i32 --
            "haxe_bytes_get_int32" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let buf = read_bytes_slice(&mut caller, h, pos, 4);
                            let val = i32::from_le_bytes(buf[0..4].try_into().unwrap());
                            if std::env::var_os("RAYZOR_WASM_TRACE_BYTES").is_some()
                                && pos <= 24
                            {
                                let header = read_haxe_bytes_header(&mut caller, h);
                                eprintln!(
                                    "[wasm-runner] bytes.getInt32 h={h:#x} pos={pos} header={header:?} buf={buf:02x?} -> {val}"
                                );
                            }
                            results[0] = ret_int(val, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt32(handle, pos, val) --
            "haxe_bytes_set_int32" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let val = val_i32(&params[2]);
                            write_bytes_slice(&mut caller, h, pos, &val.to_le_bytes());
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getInt64(handle, pos) -> i64 --
            "haxe_bytes_get_int64" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let buf = read_bytes_slice(&mut caller, h, pos, 8);
                            let val = i64::from_le_bytes(buf[0..8].try_into().unwrap());
                            results[0] = match &rt {
                                ValType::I64 => Val::I64(val),
                                _ => ret_int(val as i32, &rt),
                            };
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt64(handle, pos, val) --
            "haxe_bytes_set_int64" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let val = val_i64(&params[2]);
                            write_bytes_slice(&mut caller, h, pos, &val.to_le_bytes());
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getFloat(handle, pos) -> f32 --
            "haxe_bytes_get_float" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let buf = read_bytes_slice(&mut caller, h, pos, 4);
                            let val = f32::from_le_bytes(buf[0..4].try_into().unwrap());
                            results[0] = ret_f32(val, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setFloat(handle, pos, val) --
            "haxe_bytes_set_float" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let val = val_f32(&params[2]);
                            write_bytes_slice(&mut caller, h, pos, &val.to_le_bytes());
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getDouble(handle, pos) -> f64 --
            "haxe_bytes_get_double" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let buf = read_bytes_slice(&mut caller, h, pos, 8);
                            let val = f64::from_le_bytes(buf[0..8].try_into().unwrap());
                            results[0] = ret_f64(val, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setDouble(handle, pos, val) --
            "haxe_bytes_set_double" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let val = val_f64(&params[2]);
                            write_bytes_slice(&mut caller, h, pos, &val.to_le_bytes());
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- fill(handle, pos, len, val) --
            "haxe_bytes_fill" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            // pos/len/val are raw primitives — do NOT unbox
                            // (see haxe_bytes_sub note).
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let len = val_i32(&params[2]) as usize;
                            let val = val_i32(&params[3]) as u8;
                            fill_bytes_slice(&mut caller, h, pos, len, val);
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- blit(dest, destPos, src, srcPos, len) --
            "haxe_bytes_blit" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            // positions/len are raw primitives — do NOT unbox
                            // (see haxe_bytes_sub note).
                            let dest_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let dest_pos = val_i32(&params[1]) as usize;
                            let src_h = unbox_int_from_memory(&mut caller, val_i32(&params[2]));
                            let src_pos = val_i32(&params[3]) as usize;
                            let len = val_i32(&params[4]) as usize;
                            let src_bytes = read_bytes_slice(&mut caller, src_h, src_pos, len);
                            write_bytes_slice(&mut caller, dest_h, dest_pos, &src_bytes);
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- compare(a, b) -> i32 --
            "haxe_bytes_compare" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                            let a_len = bytes_total_len(&mut caller, a_h);
                            let b_len = bytes_total_len(&mut caller, b_h);
                            let a = read_bytes_slice(&mut caller, a_h, 0, a_len);
                            let b = read_bytes_slice(&mut caller, b_h, 0, b_len);
                            results[0] = ret_int(a.as_slice().cmp(b.as_slice()) as i32, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- subWithBase(handle, base, offLo, offHi, len) -> handle --
            //
            // Wide-offset variant of `bytes.sub` for files > 2 GiB. The
            // combined `base + (offHi << 32) + offLo` can exceed i32 range,
            // which is why GGUFReader splits the u64 offset into two i32
            // halves and passes both. Used by `tensorBytes(info)` to slice
            // each Q4_K_M / Q6_K weight tensor out of the mmapped GGUF.
            // See bugs_gguf_tensor_decode_crash for the historical fix.
            "haxe_bytes_sub_base_u64lh" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            // All five params are primitive i32 from Haxe (handle,
                            // base, offLo, offHi, len). They are NOT boxed
                            // DynamicValues, so skip unbox_int_from_memory which
                            // would mis-interpret a large primitive (e.g. a 215MB
                            // tensor length) as a Dynamic pointer and dereference
                            // garbage out of the wasm memory region holding the
                            // mmap'd GGUF file.
                            let h = val_i32(&params[0]);
                            let base = val_i32(&params[1]) as u64;
                            let off_lo = val_i32(&params[2]) as u64 & 0xFFFF_FFFF;
                            let off_hi = val_i32(&params[3]) as u64 & 0xFFFF_FFFF;
                            let len = val_i32(&params[4]) as usize;
                            let abs = base.wrapping_add(off_lo | (off_hi << 32)) as usize;
                            // wasm-memory backed source: no copy, just a new
                            // header pointing at src.data + abs.
                            if looks_like_wasm_ptr(h) {
                                if let Some((data_addr, total_len, _)) =
                                    read_haxe_bytes_header(&mut caller, h)
                                {
                                    let end = abs.saturating_add(len).min(total_len as usize);
                                    let view_len = if abs < end { end - abs } else { 0 };
                                    let header_ptr = make_haxe_bytes_view(
                                        &mut caller,
                                        data_addr + abs as u32,
                                        view_len as u32,
                                    );
                                    results[0] = ret_ptr(header_ptr, &rt);
                                    return Ok(());
                                }
                            }
                            // Legacy host-side path: materialize a fresh copy
                            // in wasm memory so the new header is usable by
                            // runtime-wasm.
                            let sub = caller
                                .data()
                                .bytes_handles
                                .get(&h)
                                .map(|v| {
                                    let end = abs.saturating_add(len).min(v.len());
                                    if abs < end {
                                        v[abs..end].to_vec()
                                    } else {
                                        vec![0u8; len]
                                    }
                                })
                                .unwrap_or_else(|| vec![0u8; len]);
                            let header_ptr = write_haxe_bytes_in_wasm(&mut caller, &sub);
                            results[0] = ret_ptr(header_ptr, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- toString(handle) -> HaxeString pointer in WASM memory --
            // Reads the bytes from the host-side bytes_handles map, UTF-8-decodes
            // (lossy — invalid sequences become U+FFFD), and writes a HaxeString
            // {data_ptr, len, cap} into WASM linear memory using the host bump
            // allocator. Returns the HaxeString struct pointer as i32.
            "haxe_bytes_to_string" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let total = bytes_total_len(&mut caller, h);
                            let bytes = read_bytes_slice(&mut caller, h, 0, total);
                            let s = String::from_utf8_lossy(&bytes).into_owned();
                            let ptr = write_haxe_string(&mut caller, &s);
                            results[0] = ret_ptr(ptr, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- sub(handle, pos, len) -> handle --
            "haxe_bytes_sub" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            // pos/len are raw primitive integers (file offsets /
                            // lengths), NEVER boxed Dynamics. Running them through
                            // unbox_int_from_memory mis-derefs large 4-aligned
                            // offsets that happen to point at bytes resembling a
                            // DynamicValue header, reading a tensor NAME from the
                            // wrong place during GGUF load — the tensor then
                            // registers under a corrupt key and surfaces later as
                            // a heap-layout-dependent "missing weight 'blk.N...'"
                            // roulette. Same rule as haxe_bytes_sub_base_u64lh.
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let pos = val_i32(&params[1]) as usize;
                            let len = val_i32(&params[2]) as usize;
                            // wasm-memory backed source: no copy, just a new
                            // header pointing at src.data + pos.
                            if looks_like_wasm_ptr(h) {
                                if let Some((data_addr, total_len, _)) =
                                    read_haxe_bytes_header(&mut caller, h)
                                {
                                    let end = pos.saturating_add(len).min(total_len as usize);
                                    let view_len = if pos < end { end - pos } else { 0 };
                                    let header_ptr = make_haxe_bytes_view(
                                        &mut caller,
                                        data_addr + pos as u32,
                                        view_len as u32,
                                    );
                                    results[0] = ret_ptr(header_ptr, &rt);
                                    return Ok(());
                                }
                            }
                            let sub = caller
                                .data()
                                .bytes_handles
                                .get(&h)
                                .map(|v| {
                                    let end = (pos + len).min(v.len());
                                    if pos < end {
                                        v[pos..end].to_vec()
                                    } else {
                                        vec![0u8; len]
                                    }
                                })
                                .unwrap_or_else(|| vec![0u8; len]);
                            let header_ptr = write_haxe_bytes_in_wasm(&mut caller, &sub);
                            results[0] = ret_ptr(header_ptr, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            _ => continue,
        }

        registered.insert(name.clone());
    }

    // -- Register File.getBytes() host stub --
    //
    // `File.getBytes(path: String) -> haxe.io.Bytes` is the entrypoint Haxe
    // code uses to slurp a file into a Bytes handle. The wasm path can't go
    // through WASI because preopened-dir sandboxing only covers the cwd
    // subtree (Llama GGUFs live elsewhere). The host stub reads through
    // `std::fs::read` directly — same trust boundary as `rayzor run --wasm`
    // already implies (the user picked the source program).
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        if name != "haxe_file_get_bytes" {
            continue;
        }
        let rt = func_ty.results().next().unwrap_or(ValType::I32);
        linker
            .func_new(
                "rayzor",
                name,
                func_ty.clone(),
                move |mut caller, params, results| {
                    let path = read_haxe_string(&mut caller, val_i32(&params[0]));
                    let data = std::fs::read(&path).unwrap_or_else(|e| {
                        eprintln!(
                            "[wasm-runner] haxe_file_get_bytes: read {:?} failed: {}",
                            path, e
                        );
                        Vec::new()
                    });
                    // Allocate the file contents + HaxeBytes header in WASM
                    // linear memory so the runtime-wasm cdylib can dereference
                    // the handle as `*const HaxeBytes`.
                    let header_ptr = write_haxe_bytes_in_wasm(&mut caller, &data);
                    if std::env::var_os("RAYZOR_WASM_TRACE_BYTES").is_some() {
                        let prefix_len = data.len().min(24);
                        eprintln!(
                            "[wasm-runner] haxe_file_get_bytes path={path:?} len={} header={header_ptr:#x} first={:02x?}",
                            data.len(),
                            &data[..prefix_len]
                        );
                    }
                    results[0] = ret_ptr(header_ptr, &rt);
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register {}: {}", name, e))?;
        registered.insert(name.clone());
    }

    // -- Register Sys.args() host stub --
    //
    // Sys.args() lowers to a `haxe_sys_args` import that should return a
    // `Array<String>` of the CLI tail args (everything after `--` on the
    // rayzor invocation). We build the HaxeArray + each HaxeString inside
    // WASM linear memory using the host bump allocator so the wasm side can
    // index it through the standard array_get_i64 / arr.length paths.
    //
    // HaxeArray layout (32 bytes — MIR's i64-stride struct):
    //   offset  0: data_ptr (u32)  + 4 unused
    //   offset  8: len      (u32)  + 4 unused
    //   offset 16: cap      (u32)  + 4 unused
    //   offset 24: elem_size(u32)  + 4 unused (set to 8 for i64-pointer stride)
    // The data block is `len * 8` bytes; each slot holds a HaxeString pointer
    // (i32 wasm address) zero-extended into an i64.
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        if name != "haxe_sys_args" {
            continue;
        }
        let rt = func_ty.results().next().unwrap_or(ValType::I32);
        linker
            .func_new(
                "rayzor",
                name,
                func_ty.clone(),
                move |mut caller, _params, results| {
                    let args = caller.data().program_args.clone();
                    let n = args.len() as u32;
                    // Allocate per-string HaxeStruct pointers first; we need them
                    // before we can write the data block.
                    let mut string_ptrs: Vec<i32> = Vec::with_capacity(n as usize);
                    for s in &args {
                        string_ptrs.push(write_haxe_string(&mut caller, s));
                    }
                    // Data block: n i64 slots, each holding one HaxeString ptr.
                    let data_bytes = (n as u32) * 8;
                    let data_addr = host_alloc(&mut caller, data_bytes.max(1));
                    for (i, &sptr) in string_ptrs.iter().enumerate() {
                        let off = data_addr + (i as u32) * 8;
                        // Low 4 bytes: pointer; high 4 bytes: zero.
                        write_wasm_mem(&mut caller, off, &(sptr as u32).to_le_bytes());
                        write_wasm_mem(&mut caller, off + 4, &0u32.to_le_bytes());
                    }
                    // HaxeArray header: 32 bytes (i64-stride fields).
                    let header_addr = host_alloc(&mut caller, 32);
                    write_wasm_mem(&mut caller, header_addr, &data_addr.to_le_bytes());
                    write_wasm_mem(&mut caller, header_addr + 4, &0u32.to_le_bytes());
                    write_wasm_mem(&mut caller, header_addr + 8, &n.to_le_bytes());
                    write_wasm_mem(&mut caller, header_addr + 12, &0u32.to_le_bytes());
                    write_wasm_mem(&mut caller, header_addr + 16, &n.to_le_bytes());
                    write_wasm_mem(&mut caller, header_addr + 20, &0u32.to_le_bytes());
                    write_wasm_mem(&mut caller, header_addr + 24, &8u32.to_le_bytes());
                    write_wasm_mem(&mut caller, header_addr + 28, &0u32.to_le_bytes());
                    results[0] = ret_ptr(header_addr as i32, &rt);
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register {}: {}", name, e))?;
        registered.insert(name.clone());
    }

    // -- Register typed Dynamic boxing host stub --
    //
    // Generic Optional<T> lowering uses this helper after monomorphization
    // fills in the concrete type tag. The runtime-wasm cdylib does not yet
    // export it, so leaving it to the generic zero stub makes nullable
    // generic values collapse to null. GGUF metadata uses StringMap<MetaValue>,
    // which exercises that path during lookup.
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        if name != "haxe_box_typed_ptr" {
            continue;
        }
        let rt = func_ty.results().next().unwrap_or(ValType::I32);
        linker
            .func_new(
                "rayzor",
                name,
                func_ty.clone(),
                move |mut caller, params, results| {
                    let value = match params.first() {
                        Some(Val::I64(x)) => *x as u64,
                        Some(Val::I32(x)) => *x as u32 as u64,
                        Some(Val::F64(x)) => *x,
                        Some(Val::F32(x)) => *x as u64,
                        _ => 0,
                    };
                    let type_tag = params.get(1).map(val_i32).unwrap_or(0);
                    let out_ptr = match type_tag {
                        // Tags match runtime/src/type_system.rs:
                        // 1=Int, 2=Bool, 4=Float, 5=String, 6=Reference.
                        1 => box_int_quiet_in_wasm(&mut caller, value as i32),
                        2 => box_bool_quiet_in_wasm(&mut caller, value != 0),
                        4 => box_float_in_wasm(&mut caller, f64::from_bits(value)),
                        5 => {
                            if value == 0 {
                                box_dynamic_header_in_wasm(&mut caller, 1, 0)
                            } else {
                                box_dynamic_header_in_wasm(&mut caller, 5, value as u32)
                            }
                        }
                        6 => {
                            if value == 0 {
                                box_dynamic_header_in_wasm(&mut caller, 1, 0)
                            } else {
                                box_dynamic_header_in_wasm(&mut caller, 6, value as u32)
                            }
                        }
                        _ => box_int_quiet_in_wasm(&mut caller, value as i32),
                    };
                    if std::env::var_os("RAYZOR_WASM_TRACE_DYNAMIC").is_some() {
                        eprintln!(
                            "[wasm-runner] haxe_box_typed_ptr value={:#x} tag={} -> {:#x}",
                            value, type_tag, out_ptr
                        );
                    }
                    results[0] = match rt {
                        ValType::I64 => Val::I64(out_ptr as u32 as i64),
                        _ => Val::I32(out_ptr),
                    };
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register {}: {}", name, e))?;
        registered.insert(name.clone());
    }

    // -- Register Haxe StringMap host stubs --
    //
    // GGUFReader parses the metadata KV table into a `StringMap<MetaValue>`
    // via `meta.set(key, value)`, then GGUFLoader.metadataFromReader reads
    // it back through `meta.get(key)` / `meta.exists(key)`. Without these
    // four, every metadata read returns a null handle and the program traps
    // inside the enum switch.
    //
    // Storage: `stringmap_handles: BTreeMap<i32, BTreeMap<String, i64>>` on
    // WasmState. Each new() allocates a fresh map and returns its small-int
    // handle. set/get/exists key by reading the HaxeString from wasm memory
    // (via `read_haxe_string`). Values are passed as i64 because the MIR
    // emits all calls through the i64-stride generic V slot — they end up
    // holding wasm-side pointers, primitive ints, or boxed Dynamic.
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        let kind = match name.as_str() {
            "haxe_stringmap_new" => "new",
            "haxe_stringmap_set" => "set",
            "haxe_stringmap_get" => "get",
            "haxe_stringmap_exists" => "exists",
            "haxe_stringmap_remove" => "remove",
            "haxe_stringmap_keys" => "keys",
            _ => continue,
        };
        let rt = func_ty.results().next().unwrap_or(ValType::I32);
        let func_ty_clone = func_ty.clone();
        linker
            .func_new(
                "rayzor",
                name,
                func_ty_clone,
                move |mut caller, params, results| {
                    match kind {
                        "new" => {
                            let id = {
                                let s = caller.data_mut();
                                let id = s.next_stringmap_id;
                                s.next_stringmap_id += 1;
                                s.stringmap_handles.insert(id, BTreeMap::new());
                                id
                            };
                            if std::env::var_os("RAYZOR_WASM_TRACE_STRINGMAP").is_some() {
                                eprintln!("[wasm-runner] stringmap.new -> h={id}");
                            }
                            results[0] = ret_int(id, &rt);
                        }
                        "set" => {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let key = read_haxe_string(&mut caller, val_i32(&params[1]));
                            // Value param may be i32 or i64 depending on the
                            // emitted call shape. Promote either way to i64.
                            let v = if params.len() >= 3 {
                                match &params[2] {
                                    Val::I32(x) => *x as i64,
                                    Val::I64(x) => *x,
                                    Val::F32(_) | Val::F64(_) => 0,
                                    _ => 0,
                                }
                            } else {
                                0
                            };
                            if std::env::var_os("RAYZOR_WASM_TRACE_STRINGMAP").is_some() {
                                eprintln!("[wasm-runner] stringmap.set h={h} key={key:?} v={v:#x}");
                            }
                            if let Some(map) = caller.data_mut().stringmap_handles.get_mut(&h) {
                                map.insert(key, v);
                            }
                        }
                        "get" => {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let key = read_haxe_string(&mut caller, val_i32(&params[1]));
                            let val = caller
                                .data()
                                .stringmap_handles
                                .get(&h)
                                .and_then(|m| m.get(&key))
                                .copied()
                                .unwrap_or(0);
                            if std::env::var_os("RAYZOR_WASM_TRACE_STRINGMAP").is_some() {
                                eprintln!(
                                    "[wasm-runner] stringmap.get h={h} key={key:?} -> {val:#x}"
                                );
                            }
                            // Return either i32 or i64 depending on the
                            // declared result type.
                            match rt {
                                ValType::I64 => results[0] = Val::I64(val),
                                _ => results[0] = Val::I32(val as i32),
                            }
                        }
                        "exists" => {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let key = read_haxe_string(&mut caller, val_i32(&params[1]));
                            let yes = caller
                                .data()
                                .stringmap_handles
                                .get(&h)
                                .map(|m| m.contains_key(&key))
                                .unwrap_or(false);
                            if std::env::var_os("RAYZOR_WASM_TRACE_STRINGMAP").is_some() {
                                eprintln!(
                                    "[wasm-runner] stringmap.exists h={h} key={key:?} -> {yes}"
                                );
                            }
                            results[0] = ret_int(if yes { 1 } else { 0 }, &rt);
                        }
                        "remove" => {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let key = read_haxe_string(&mut caller, val_i32(&params[1]));
                            let removed = caller
                                .data_mut()
                                .stringmap_handles
                                .get_mut(&h)
                                .map(|m| m.remove(&key).is_some())
                                .unwrap_or(false);
                            if std::env::var_os("RAYZOR_WASM_TRACE_STRINGMAP").is_some() {
                                eprintln!(
                                    "[wasm-runner] stringmap.remove h={h} key={key:?} -> {removed}"
                                );
                            }
                            results[0] = ret_int(if removed { 1 } else { 0 }, &rt);
                        }
                        "keys" => {
                            // Returns an Array<String> of the map's keys. We
                            // build it the same way `haxe_sys_args` does.
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let keys: Vec<String> = caller
                                .data()
                                .stringmap_handles
                                .get(&h)
                                .map(|m| m.keys().cloned().collect())
                                .unwrap_or_default();
                            let n = keys.len() as u32;
                            let mut string_ptrs: Vec<i32> = Vec::with_capacity(n as usize);
                            for k in &keys {
                                string_ptrs.push(write_haxe_string(&mut caller, k));
                            }
                            let data_addr = host_alloc(&mut caller, (n * 8).max(1));
                            for (i, &sp) in string_ptrs.iter().enumerate() {
                                let off = data_addr + (i as u32) * 8;
                                write_wasm_mem(&mut caller, off, &(sp as u32).to_le_bytes());
                                write_wasm_mem(&mut caller, off + 4, &0u32.to_le_bytes());
                            }
                            let header = host_alloc(&mut caller, 32);
                            write_wasm_mem(&mut caller, header, &data_addr.to_le_bytes());
                            write_wasm_mem(&mut caller, header + 4, &0u32.to_le_bytes());
                            write_wasm_mem(&mut caller, header + 8, &n.to_le_bytes());
                            write_wasm_mem(&mut caller, header + 12, &0u32.to_le_bytes());
                            write_wasm_mem(&mut caller, header + 16, &n.to_le_bytes());
                            write_wasm_mem(&mut caller, header + 20, &0u32.to_le_bytes());
                            write_wasm_mem(&mut caller, header + 24, &8u32.to_le_bytes());
                            write_wasm_mem(&mut caller, header + 28, &0u32.to_le_bytes());
                            results[0] = ret_ptr(header as i32, &rt);
                        }
                        _ => {}
                    }
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register {}: {}", name, e))?;
        registered.insert(name.clone());
    }

    // -- Register String byte-codec host stubs --
    //
    // BPETokenizer's byteEncoderTable + byteDecode rely on the
    // codepoint↔string round-trip Haxe's String exposes via
    // `String.fromCharCode(c)` and `s.charCodeAt(i)`. Rayzor strings are
    // UTF-8 byte arrays, so charCodeAt returns a raw byte (0..255) — see
    // `bugs_bpe_utf8_codepoint_walk` for the historic native equivalence.
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        let kind = match name.as_str() {
            "haxe_string_char_code_at" => "char_code_at",
            "haxe_string_char_code_at_ptr" => "char_code_at",
            "haxe_string_from_char_code" => "from_char_code",
            _ => continue,
        };
        let rt = func_ty.results().next().unwrap_or(ValType::I32);
        linker
            .func_new(
                "rayzor",
                name,
                func_ty.clone(),
                move |mut caller, params, results| {
                    match kind {
                        "char_code_at" => {
                            let s = read_haxe_string(&mut caller, val_i32(&params[0]));
                            let idx = val_i32(&params[1]) as usize;
                            let val = s.as_bytes().get(idx).copied().unwrap_or(0) as i32;
                            results[0] = ret_int(val, &rt);
                        }
                        "from_char_code" => {
                            let code = val_i32(&params[0]);
                            // Native Haxe semantics: encode the codepoint as
                            // UTF-8 bytes. For ASCII (0..127) the result is a
                            // single byte; for higher codepoints it's 2-4 bytes.
                            let s = match char::from_u32(code as u32) {
                                Some(c) => c.to_string(),
                                None => String::new(),
                            };
                            let ptr = write_haxe_string(&mut caller, &s);
                            results[0] = ret_ptr(ptr, &rt);
                        }
                        _ => {}
                    }
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register {}: {}", name, e))?;
        registered.insert(name.clone());
    }

    // -- Register Std type-conversion host stubs --
    //
    // `Std.int(f)` truncates Float→Int. `Std.parseInt` / `Std.parseFloat`
    // parse strings — used by Main.hx CLI parsing. `Std.string(x)` lets
    // the benchmarking helpers format floats as text.
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        let kind = match name.as_str() {
            "haxe_std_int" => "int",
            "haxe_std_parse_int" => "parse_int",
            "haxe_std_parse_float" => "parse_float",
            "haxe_std_string" => "string",
            "haxe_std_string_ptr" => "string_ptr",
            _ => continue,
        };
        let rt = func_ty.results().next().unwrap_or(ValType::I32);
        linker
            .func_new(
                "rayzor",
                name,
                func_ty.clone(),
                move |mut caller, params, results| {
                    match kind {
                        "int" => {
                            // Param may be f64 or boxed Float. wasmtime's
                            // `Val::F64(u64)` / `Val::F32(u32)` carry the
                            // raw BIT PATTERN, not the float value — use
                            // `from_bits` to recover the float before
                            // truncating to i32. Otherwise `Std.int(64.0)`
                            // reads 0x4050_0000_0000_0000's low 32 bits = 0.
                            let v = match &params[0] {
                                Val::F64(x) => f64::from_bits(*x) as i32,
                                Val::F32(x) => f32::from_bits(*x) as i32,
                                Val::I32(x) => *x,
                                Val::I64(x) => *x as i32,
                                _ => 0,
                            };
                            results[0] = ret_int(v, &rt);
                        }
                        "parse_int" => {
                            let s = read_haxe_string(&mut caller, val_i32(&params[0]));
                            let trimmed = s.trim();
                            let parsed: i32 = trimmed.parse::<i64>().map(|v| v as i32).unwrap_or(0);
                            results[0] = ret_int(parsed, &rt);
                        }
                        "parse_float" => {
                            let s = read_haxe_string(&mut caller, val_i32(&params[0]));
                            let v: f64 = s.trim().parse().unwrap_or(f64::NAN);
                            // Result type is always F64 for parse_float.
                            results[0] = Val::F64(v.to_bits());
                        }
                        "string" => {
                            // Polymorphic over input; for the load+decode path
                            // the most common callsite is Std.string(Float).
                            // Same Val::F64/F32 bit-pattern caveat as `int`
                            // above — decode via from_bits before printing.
                            let formatted = match &params[0] {
                                Val::F64(x) => format!("{}", f64::from_bits(*x)),
                                Val::F32(x) => format!("{}", f32::from_bits(*x)),
                                Val::I32(x) => format!("{}", x),
                                Val::I64(x) => format!("{}", x),
                                _ => String::new(),
                            };
                            let ptr = write_haxe_string(&mut caller, &formatted);
                            results[0] = ret_ptr(ptr, &rt);
                        }
                        "string_ptr" => {
                            let formatted = format_std_string_ptr(&mut caller, val_i32(&params[0]));
                            let ptr = write_haxe_string(&mut caller, &formatted);
                            results[0] = ret_ptr(ptr, &rt);
                        }
                        _ => {}
                    }
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register {}: {}", name, e))?;
        registered.insert(name.clone());
    }

    // -- Register Sys host stubs --
    //
    // `Sys.getEnv` is the env-var gate LlamaArch.build + GGUFTokenizer use
    // to opt into RAYZOR_KV_Q8 et al. `Sys.time` is the wall-clock the
    // benchmark prints. `Sys.println` writes diagnostic trace lines.
    // `Sys.exit` aborts on bad CLI args.
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        let kind = match name.as_str() {
            "haxe_sys_get_env" => "get_env",
            "haxe_sys_time" => "time",
            "haxe_sys_println" => "println",
            "haxe_sys_exit" => "exit",
            _ => continue,
        };
        let rt = func_ty.results().next().unwrap_or(ValType::I32);
        linker
            .func_new(
                "rayzor",
                name,
                func_ty.clone(),
                move |mut caller, params, results| {
                    match kind {
                        "get_env" => {
                            let key = read_haxe_string(&mut caller, val_i32(&params[0]));
                            match std::env::var(&key) {
                                Ok(v) => {
                                    let ptr = write_haxe_string(&mut caller, &v);
                                    results[0] = ret_ptr(ptr, &rt);
                                }
                                Err(_) => {
                                    // Null pointer — Haxe's `Sys.getEnv` returns
                                    // null when the var is absent.
                                    results[0] = ret_int(0, &rt);
                                }
                            }
                        }
                        "time" => {
                            use std::time::{SystemTime, UNIX_EPOCH};
                            let secs = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_secs_f64())
                                .unwrap_or(0.0);
                            results[0] = Val::F64(secs.to_bits());
                        }
                        "println" => {
                            let s = read_haxe_string(&mut caller, val_i32(&params[0]));
                            println!("{}", s);
                        }
                        "exit" => {
                            let code = val_i32(&params[0]);
                            std::process::exit(code);
                        }
                        _ => {}
                    }
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to register {}: {}", name, e))?;
        registered.insert(name.clone());
    }

    // -- Register EReg host functions --
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        let is_ereg = matches!(
            name.as_str(),
            "haxe_ereg_new"
                | "haxe_ereg_match"
                | "haxe_ereg_matched"
                | "haxe_ereg_matched_left"
                | "haxe_ereg_matched_right"
                | "haxe_ereg_matched_pos"
                | "haxe_ereg_matched_pos_anon"
                | "haxe_ereg_match_sub"
                | "haxe_ereg_replace"
                | "haxe_ereg_escape"
                | "haxe_ereg_split"
                | "haxe_ereg_map"
        );
        if !is_ereg {
            continue;
        }

        match name.as_str() {
            // new(pattern, flags) -> handle
            "haxe_ereg_new" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let pattern = read_haxe_string(&mut caller, val_i32(&params[0]));
                            let flags = read_haxe_string(&mut caller, val_i32(&params[1]));
                            let mut re_pattern = pattern.clone();
                            // Convert Haxe regex flags to Rust regex flags
                            let case_insensitive = flags.contains('i');
                            let multiline = flags.contains('m');
                            let dotall = flags.contains('s');
                            if case_insensitive || multiline || dotall {
                                let mut prefix = String::from("(?");
                                if case_insensitive {
                                    prefix.push('i');
                                }
                                if multiline {
                                    prefix.push('m');
                                }
                                if dotall {
                                    prefix.push('s');
                                }
                                prefix.push(')');
                                re_pattern = format!("{}{}", prefix, re_pattern);
                            }
                            let regex = match regex::Regex::new(&re_pattern) {
                                Ok(r) => r,
                                Err(_) => {
                                    results[0] = Val::I32(0);
                                    return Ok(());
                                }
                            };
                            let s = caller.data_mut();
                            let id = s.next_ereg_id;
                            s.next_ereg_id += 1;
                            s.ereg_handles.insert(
                                id,
                                ERegState {
                                    pattern,
                                    flags,
                                    regex,
                                    last_input: None,
                                    last_match: None,
                                    last_captures: vec![],
                                    match_start: 0,
                                    match_end: 0,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // match(this, s) -> bool (boxed)
            "haxe_ereg_match" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let s = read_haxe_string(&mut caller, val_i32(&params[1]));
                            let matched = {
                                if let Some(st) = caller.data_mut().ereg_handles.get_mut(&h) {
                                    if let Some(caps) = st.regex.captures(&s) {
                                        st.match_start =
                                            caps.get(0).map(|m| m.start()).unwrap_or(0);
                                        st.match_end = caps.get(0).map(|m| m.end()).unwrap_or(0);
                                        st.last_captures = (0..caps.len())
                                            .map(|i| caps.get(i).map(|m| m.as_str().to_string()))
                                            .collect();
                                        st.last_input = Some(s);
                                        true
                                    } else {
                                        st.last_input = Some(s);
                                        st.last_captures.clear();
                                        false
                                    }
                                } else {
                                    false
                                }
                            };
                            results[0] = Val::I32(if matched { 1 } else { 0 });
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // matched(this, n) -> String
            "haxe_ereg_matched" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let n =
                                unbox_int_from_memory(&mut caller, val_i32(&params[1])) as usize;
                            let val = caller
                                .data()
                                .ereg_handles
                                .get(&h)
                                .and_then(|st| st.last_captures.get(n).cloned().flatten())
                                .unwrap_or_default();
                            let ptr = write_haxe_string(&mut caller, &val);
                            results[0] = Val::I32(ptr);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // matchedLeft(this) -> String
            "haxe_ereg_matched_left" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let val = caller
                                .data()
                                .ereg_handles
                                .get(&h)
                                .and_then(|st| {
                                    st.last_input
                                        .as_ref()
                                        .map(|s| s[..st.match_start].to_string())
                                })
                                .unwrap_or_default();
                            let ptr = write_haxe_string(&mut caller, &val);
                            results[0] = Val::I32(ptr);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // matchedRight(this) -> String
            "haxe_ereg_matched_right" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let val = caller
                                .data()
                                .ereg_handles
                                .get(&h)
                                .and_then(|st| {
                                    st.last_input
                                        .as_ref()
                                        .map(|s| s[st.match_end..].to_string())
                                })
                                .unwrap_or_default();
                            let ptr = write_haxe_string(&mut caller, &val);
                            results[0] = Val::I32(ptr);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // replace(this, s, by) -> String
            "haxe_ereg_replace" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let s = read_haxe_string(&mut caller, val_i32(&params[1]));
                            let by = read_haxe_string(&mut caller, val_i32(&params[2]));
                            let replaced = caller
                                .data()
                                .ereg_handles
                                .get(&h)
                                .map(|st| {
                                    if st.flags.contains('g') {
                                        st.regex.replace_all(&s, by.as_str()).to_string()
                                    } else {
                                        st.regex.replace(&s, by.as_str()).to_string()
                                    }
                                })
                                .unwrap_or(s);
                            let ptr = write_haxe_string(&mut caller, &replaced);
                            results[0] = Val::I32(ptr);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // static escape(s) -> String
            "haxe_ereg_escape" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let s = read_haxe_string(&mut caller, val_i32(&params[0]));
                            let escaped = regex::escape(&s);
                            let ptr = write_haxe_string(&mut caller, &escaped);
                            results[0] = Val::I32(ptr);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // matchSub, matchedPos, split, map — return stubs for now
            _ => {
                let results_tys: Vec<ValType> = func_ty.results().collect();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, _params, out| {
                            for (i, r) in results_tys.iter().enumerate() {
                                out[i] = match r {
                                    ValType::I64 => Val::I64(0),
                                    ValType::F32 => Val::F32(0),
                                    ValType::F64 => Val::F64(0),
                                    _ => Val::I32(0),
                                };
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }
        }
        registered.insert(name.clone());
    }

    // -- Register Mutex/Arc/Box host functions --
    fn canonical_sync_name(name: &str) -> Option<&str> {
        match name {
            // Qualified names
            "rayzor_mutex_init"
            | "rayzor_mutex_lock"
            | "rayzor_mutex_try_lock"
            | "rayzor_mutex_is_locked"
            | "rayzor_mutex_unlock"
            | "rayzor_mutex_guard_get"
            | "mutex_guard_unlock"
            | "MutexGuard_unlock"
            | "rayzor_arc_init"
            | "rayzor_arc_clone"
            | "rayzor_arc_get"
            | "rayzor_arc_as_ptr"
            | "rayzor_arc_try_unwrap"
            | "rayzor_arc_strong_count"
            | "rayzor_box_init"
            | "rayzor_box_unbox"
            | "rayzor_box_raw"
            | "rayzor_box_free" => Some(name),
            // Bare names from runtime-wasm (may appear as camelCase or snake_case)
            "lock" => Some("rayzor_mutex_lock"),
            "unlock" | "MutexGuard_unlock" | "mutex_guard_unlock" => Some("rayzor_mutex_unlock"),
            "isLocked" | "is_locked" => Some("rayzor_mutex_is_locked"),
            "tryLock" | "try_lock" => Some("rayzor_mutex_try_lock"),
            "guard_get" => Some("rayzor_mutex_guard_get"),
            _ => None,
        }
    }
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        let canon = match canonical_sync_name(name) {
            Some(c) => c,
            None => continue,
        };

        match canon {
            // -- rayzor_mutex_init(val) -> handle (raw, NOT boxed) --
            "rayzor_mutex_init" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let val = val_i32(&params[0]);
                            let s = caller.data_mut();
                            let id = s.next_mutex_id;
                            s.next_mutex_id += 1;
                            s.mutex_handles.insert(
                                id,
                                MutexState {
                                    locked: false,
                                    value: val,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_lock(handle) -> raw guard handle --
            "rayzor_mutex_lock" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            if let Some(st) = caller.data_mut().mutex_handles.get_mut(&h) {
                                st.locked = true;
                            }
                            results[0] = Val::I32(h);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_try_lock(handle) -> raw 1 if acquired, raw 0 if already locked --
            "rayzor_mutex_try_lock" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let acquired = {
                                if let Some(st) = caller.data_mut().mutex_handles.get_mut(&h) {
                                    if !st.locked {
                                        st.locked = true;
                                        1
                                    } else {
                                        0
                                    }
                                } else {
                                    0
                                }
                            };
                            results[0] = Val::I32(acquired);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_is_locked(handle) -> raw 1 if locked, raw 0 if not --
            "rayzor_mutex_is_locked" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let locked = caller
                                .data()
                                .mutex_handles
                                .get(&h)
                                .map(|st| if st.locked { 1 } else { 0 })
                                .unwrap_or(0);
                            results[0] = Val::I32(locked);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_unlock(handle) -> void --
            "rayzor_mutex_unlock" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            if let Some(st) = caller.data_mut().mutex_handles.get_mut(&h) {
                                st.locked = false;
                            }
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_guard_get(handle) -> raw value --
            "rayzor_mutex_guard_get" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let val = caller
                                .data()
                                .mutex_handles
                                .get(&h)
                                .map(|st| st.value)
                                .unwrap_or(0);
                            results[0] = Val::I32(val);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- Arc: identity pass-through --
            "rayzor_arc_init"
            | "rayzor_arc_clone"
            | "rayzor_arc_get"
            | "rayzor_arc_as_ptr"
            | "rayzor_arc_try_unwrap" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, params, results| {
                            results[0] = params[0].clone();
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_arc_strong_count -> raw 1 --
            "rayzor_arc_strong_count" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, _params, results| {
                            results[0] = Val::I32(1);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- Box: identity pass-through --
            "rayzor_box_init" | "rayzor_box_unbox" | "rayzor_box_raw" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, params, results| {
                            results[0] = params[0].clone();
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_box_free -> no-op --
            "rayzor_box_free" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, _params, results| {
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            _ => continue,
        }
        registered.insert(name.clone());
    }

    // -- Register Thread host functions --
    //
    // spawn() dispatches the closure to a persistent pool of real OS threads,
    // each running its own Store+Instance of the module over the one shared
    // linear memory (built lazily on first spawn — see `build_worker_pool`).
    // join() blocks on the closure's completion slot; the boxing variant matches
    // the native Thread<T>.join() contract, join_void() is the barrier-only path
    // the in-guest parallel matmul uses. This mirrors the browser's Web Worker
    // pool (rayzor_worker.js), where each worker is likewise its own instance
    // sharing `env.memory`.
    fn canonical_thread_name(name: &str) -> Option<&str> {
        match name {
            "rayzor_thread_spawn" => Some("rayzor_thread_spawn"),
            "rayzor_thread_join" => Some("rayzor_thread_join"),
            "rayzor_thread_join_void" => Some("rayzor_thread_join_void"),
            "rayzor_thread_is_finished" => Some("rayzor_thread_is_finished"),
            "rayzor_thread_yield_now" => Some("rayzor_thread_yield_now"),
            "rayzor_thread_sleep" => Some("rayzor_thread_sleep"),
            "rayzor_thread_current_id" => Some("rayzor_thread_current_id"),
            _ => None,
        }
    }
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        let canon = match canonical_thread_name(name) {
            Some(c) => c,
            None => continue,
        };
        let ret_ty: ValType = func_ty.results().next().unwrap_or(ValType::I32);

        match canon {
            // spawn(fn_idx, env_ptr) -> thread_handle
            // The Thread_spawn MIR wrapper extracts fn_idx from closure+0 and
            // env_ptr from closure+8 and passes them to us. We queue a pending
            // task and return a fresh thread id; the closure body runs during
            // the next join()/is_finished() call.
            "rayzor_thread_spawn" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let fn_idx = val_i32(&params[0]) as u32;
                            let env_ptr = val_i32(&params[1]);

                            // Lazily build the persistent worker pool on the
                            // first spawn. Each worker instantiates the module
                            // once (sharing this linear memory) and reserves its
                            // own stack, so dispatch is just a channel send + an
                            // indirect call — no per-spawn instantiation.
                            if caller.data().worker_pool.is_none() {
                                let engine = caller.data().engine.clone();
                                let module = caller.data().module.clone();
                                if let Some(shared) = caller.data().shared_memory.clone() {
                                    // Size the pool to the host's parallelism (so
                                    // the in-guest matmul's bands each land on a
                                    // distinct worker), capped to keep memory and
                                    // scheduling overhead bounded.
                                    let n = std::thread::available_parallelism()
                                        .map(|p| p.get())
                                        .unwrap_or(4)
                                        .clamp(2, 8);
                                    let pool = build_worker_pool(n, &engine, &module, &shared);
                                    caller.data_mut().worker_pool = Some(pool);
                                }
                            }

                            let id = {
                                let s = caller.data_mut();
                                let id = s.next_thread_id;
                                s.next_thread_id += 1;
                                id
                            };

                            // Dispatch the closure to the next pool worker
                            // (round-robin) and record its completion slot.
                            let slot = std::sync::Arc::new(JobSlot {
                                done: std::sync::Mutex::new(None),
                                cv: std::sync::Condvar::new(),
                            });
                            let dispatched =
                                if let Some(pool) = caller.data_mut().worker_pool.as_mut() {
                                    let w = pool.next;
                                    pool.next = (pool.next + 1) % pool.senders.len();
                                    pool.senders[w]
                                        .send(Job {
                                            fn_idx,
                                            env_ptr,
                                            slot: slot.clone(),
                                        })
                                        .is_ok()
                                } else {
                                    false
                                };
                            if !dispatched {
                                // No pool (no shared memory, or a closed worker):
                                // the closure can't run, so resolve to 0 rather
                                // than leave a joiner blocked forever.
                                eprintln!(
                                    "[wasm-runner] rayzor_thread_spawn: no worker pool; \
                                     closure not run"
                                );
                                publish(&slot, 0);
                            }

                            caller
                                .data_mut()
                                .thread_handles
                                .insert(id, ThreadState { slot, result: None });
                            results[0] = ret_int(id, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // join(handle) -> boxed result
            // Runs any pending thread work, then returns the cached result.
            // The result is boxed as DynamicValue* to match the native
            // rayzor_thread_join contract (Thread<T>.join() unbox path).
            "rayzor_thread_join" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = val_i32(&params[0]);
                            let result = join_worker(caller.data_mut(), h);
                            let boxed = box_int_in_wasm(&mut caller, result as i32);
                            results[0] = ret_int(boxed, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // join_void(handle) -> void
            // Blocks on the worker like rayzor_thread_join but discards its
            // result without boxing — used by the in-guest parallel matmul,
            // whose band workers return 0. Avoids the per-join box leak (the
            // matmul never unboxes a result), keeping long decodes flat.
            "rayzor_thread_join_void" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, _results| {
                            let h = val_i32(&params[0]);
                            let _ = join_worker(caller.data_mut(), h);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // is_finished(handle) -> bool
            "rayzor_thread_is_finished" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = val_i32(&params[0]);
                            let done = worker_finished(caller.data_mut(), h);
                            results[0] = ret_int(done, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // yield_now() -> void
            "rayzor_thread_yield_now" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, _params, results| {
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // sleep(ms) -> void
            "rayzor_thread_sleep" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, params, results| {
                            let ms = val_i32(&params[0]).max(0) as u64;
                            std::thread::sleep(std::time::Duration::from_millis(ms));
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // current_id() -> i64
            "rayzor_thread_current_id" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, _params, results| {
                            // Main thread = 1 (non-zero so user code can distinguish it).
                            let id = 1i64;
                            results[0] = match &rt {
                                ValType::I64 => Val::I64(id),
                                _ => ret_int(id as i32, &rt),
                            };
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            _ => continue,
        }
        registered.insert(name.clone());
    }

    // -- Register Tensor host functions --
    fn canonical_tensor_name(name: &str) -> Option<&str> {
        match name {
            "rayzor_tensor_zeros" | "Tensor_zeros" => Some("rayzor_tensor_zeros"),
            "rayzor_tensor_ones" | "Tensor_ones" => Some("rayzor_tensor_ones"),
            "rayzor_tensor_full" | "Tensor_full" => Some("rayzor_tensor_full"),
            "rayzor_tensor_from_array" | "Tensor_fromArray" | "Tensor_from_array" => {
                Some("rayzor_tensor_from_array")
            }
            "rayzor_tensor_rand" | "Tensor_rand" => Some("rayzor_tensor_rand"),
            "rayzor_tensor_ndim" => Some("rayzor_tensor_ndim"),
            "rayzor_tensor_numel" => Some("rayzor_tensor_numel"),
            "rayzor_tensor_dtype" => Some("rayzor_tensor_dtype"),
            "rayzor_tensor_get" => Some("rayzor_tensor_get"),
            "rayzor_tensor_set" => Some("rayzor_tensor_set"),
            "rayzor_tensor_reshape" => Some("rayzor_tensor_reshape"),
            "rayzor_tensor_transpose" => Some("rayzor_tensor_transpose"),
            "rayzor_tensor_permute" => Some("rayzor_tensor_permute"),
            "rayzor_tensor_slice" => Some("rayzor_tensor_slice"),
            "rayzor_tensor_add" | "Tensor_add" => Some("rayzor_tensor_add"),
            "rayzor_tensor_sub" | "Tensor_sub" => Some("rayzor_tensor_sub"),
            "rayzor_tensor_mul" | "Tensor_mul" => Some("rayzor_tensor_mul"),
            "rayzor_tensor_div" | "Tensor_div" => Some("rayzor_tensor_div"),
            "rayzor_tensor_matmul" => Some("rayzor_tensor_matmul"),
            "rayzor_tensor_dot" => Some("rayzor_tensor_dot"),
            "rayzor_tensor_sum" => Some("rayzor_tensor_sum"),
            "rayzor_tensor_mean" => Some("rayzor_tensor_mean"),
            "rayzor_tensor_max" => Some("rayzor_tensor_max"),
            "rayzor_tensor_min" => Some("rayzor_tensor_min"),
            "rayzor_tensor_sqrt" => Some("rayzor_tensor_sqrt"),
            "rayzor_tensor_exp" => Some("rayzor_tensor_exp"),
            "rayzor_tensor_log" => Some("rayzor_tensor_log"),
            "rayzor_tensor_relu" => Some("rayzor_tensor_relu"),
            "rayzor_tensor_gelu" => Some("rayzor_tensor_gelu"),
            "rayzor_tensor_silu" => Some("rayzor_tensor_silu"),
            "rayzor_tensor_softmax" => Some("rayzor_tensor_softmax"),
            "rayzor_tensor_layer_norm" => Some("rayzor_tensor_layer_norm"),
            "rayzor_tensor_rms_norm" => Some("rayzor_tensor_rms_norm"),
            "rayzor_tensor_free" => Some("rayzor_tensor_free"),
            "rayzor_tensor_data"
            | "rayzor_tensor_shape"
            | "rayzor_tensor_shape_ptr"
            | "rayzor_tensor_shape_ndim" => Some(name),
            _ => None,
        }
    }

    let runtime_tensor_exports_linked = [
        "rayzor_tensor_zeros",
        "rayzor_tensor_ones",
        "rayzor_tensor_full",
        "rayzor_tensor_from_array",
        "rayzor_tensor_numel",
        "rayzor_tensor_get",
        "rayzor_tensor_add",
        "rayzor_tensor_sum",
        "rayzor_tensor_mean",
        "rayzor_tensor_dot",
        "rayzor_tensor_matmul",
        "rayzor_tensor_reshape",
        "rayzor_tensor_sqrt",
    ]
    .iter()
    .any(|name| !rayzor_import_names.contains(*name));
    let unresolved_tensor_import_count = rayzor_imports
        .iter()
        .filter(|(name, _)| name.starts_with("rayzor_tensor_") || name.starts_with("Tensor_"))
        .count();
    if runtime_tensor_exports_linked && unresolved_tensor_import_count > 0 {
        eprintln!(
            "[wasm-runner] runtime tensor exports are linked; skipping {} tensor host fallback(s) to avoid split tensor handles",
            unresolved_tensor_import_count
        );
    }

    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        if runtime_tensor_exports_linked {
            continue;
        }
        let canon = match canonical_tensor_name(name) {
            Some(c) => c,
            None => continue,
        };
        let ret_ty: ValType = func_ty.results().next().unwrap_or(ValType::I32);

        match canon {
            // -- rayzor_tensor_zeros(dataPtr, ndim, dtype) -> handle --
            // ABI: MIR Tensor_zeros wrapper extracts (data_ptr, ndim) from
            // the shape Array<Int> via extract_array_ptr_len before the
            // extern call, so we read the shape data directly from data_ptr.
            "rayzor_tensor_zeros" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let data_ptr = val_i32(&params[0]);
                            let ndim = val_i32(&params[1]);
                            let shape = read_raw_int_array_i64stride(&mut caller, data_ptr, ndim);
                            let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(
                                id,
                                TensorState {
                                    data: vec![0.0; numel],
                                    shape,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_ones(dataPtr, ndim, dtype) -> handle --
            "rayzor_tensor_ones" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let data_ptr = val_i32(&params[0]);
                            let ndim = val_i32(&params[1]);
                            let shape = read_raw_int_array_i64stride(&mut caller, data_ptr, ndim);
                            let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(
                                id,
                                TensorState {
                                    data: vec![1.0; numel],
                                    shape,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_full(dataPtr, ndim, value, dtype) -> handle --
            "rayzor_tensor_full" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let data_ptr = val_i32(&params[0]);
                            let ndim = val_i32(&params[1]);
                            let value = val_f64(&params[2]);
                            let shape = read_raw_int_array_i64stride(&mut caller, data_ptr, ndim);
                            let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(
                                id,
                                TensorState {
                                    data: vec![value; numel],
                                    shape,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_from_array(dataPtr, len, dtype) -> handle --
            "rayzor_tensor_from_array" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let data_ptr = val_i32(&params[0]);
                            let len = val_i32(&params[1]);
                            let data = read_raw_f64_array_i64stride(&mut caller, data_ptr, len);
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(
                                id,
                                TensorState {
                                    data,
                                    shape: vec![len],
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_rand(dataPtr, ndim, dtype) -> handle --
            "rayzor_tensor_rand" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let data_ptr = val_i32(&params[0]);
                            let ndim = val_i32(&params[1]);
                            let shape = read_raw_int_array_i64stride(&mut caller, data_ptr, ndim);
                            let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
                            // Simple LCG pseudo-random for determinism in WASM
                            let mut seed: u64 = 12345;
                            let data: Vec<f64> = (0..numel)
                                .map(|_| {
                                    seed = seed
                                        .wrapping_mul(6364136223846793005)
                                        .wrapping_add(1442695040888963407);
                                    (seed >> 33) as f64 / (1u64 << 31) as f64
                                })
                                .collect();
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_ndim(handle) -> raw int --
            "rayzor_tensor_ndim" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let ndim = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| t.shape.len() as i32)
                                .unwrap_or(0);
                            results[0] = Val::I32(ndim);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_numel(handle) -> int --
            "rayzor_tensor_numel" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let numel = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| t.data.len() as i32)
                                .unwrap_or(0);
                            results[0] = Val::I32(numel);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_dtype(handle) -> raw int (0 = Float64) --
            "rayzor_tensor_dtype" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, _params, results| {
                            results[0] = Val::I32(0); // Float64 = 0
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_get(handle, indicesPtr, ndim) -> f64 --
            // ABI: indices arrive as a decomposed (data_ptr, ndim) pair from
            // the MIR Tensor_get wrapper. Multi-dim indices are flattened via
            // row-major strides over the tensor's shape.
            "rayzor_tensor_get" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let idx_ptr = val_i32(&params[1]);
                            let ndim = val_i32(&params[2]);
                            let indices = read_raw_int_array_i64stride(&mut caller, idx_ptr, ndim);
                            let val = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .and_then(|t| {
                                    let n = t.shape.len();
                                    if n == 0 {
                                        return t.data.first().copied();
                                    }
                                    let mut strides = vec![1_usize; n];
                                    for i in (0..n.saturating_sub(1)).rev() {
                                        strides[i] = strides[i + 1] * t.shape[i + 1] as usize;
                                    }
                                    let mut off = 0_usize;
                                    for (i, idx) in indices.iter().enumerate().take(n) {
                                        off += (*idx).max(0) as usize * strides[i];
                                    }
                                    t.data.get(off).copied()
                                })
                                .unwrap_or(0.0);
                            results[0] = ret_f64(val, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_set(handle, indicesPtr, ndim, value) -> void --
            "rayzor_tensor_set" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let idx_ptr = val_i32(&params[1]);
                            let ndim = val_i32(&params[2]);
                            let value = val_f64(&params[3]);
                            let indices = read_raw_int_array_i64stride(&mut caller, idx_ptr, ndim);
                            if let Some(t) = caller.data_mut().tensor_handles.get_mut(&h) {
                                let n = t.shape.len();
                                let off = if n == 0 {
                                    0
                                } else {
                                    let mut strides = vec![1_usize; n];
                                    for i in (0..n.saturating_sub(1)).rev() {
                                        strides[i] = strides[i + 1] * t.shape[i + 1] as usize;
                                    }
                                    let mut o = 0_usize;
                                    for (i, idx) in indices.iter().enumerate().take(n) {
                                        o += (*idx).max(0) as usize * strides[i];
                                    }
                                    o
                                };
                                if off < t.data.len() {
                                    t.data[off] = value;
                                }
                            }
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_reshape(handle, shapePtr, ndim) -> handle --
            "rayzor_tensor_reshape" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let shape_ptr = val_i32(&params[1]);
                            let ndim = val_i32(&params[2]);
                            let new_shape =
                                read_raw_int_array_i64stride(&mut caller, shape_ptr, ndim);
                            let data = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| t.data.clone())
                                .unwrap_or_default();
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(
                                id,
                                TensorState {
                                    data,
                                    shape: new_shape,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_transpose(handle) -> handle --
            "rayzor_tensor_transpose" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let (data, new_shape) = {
                                if let Some(t) = caller.data().tensor_handles.get(&h) {
                                    if t.shape.len() == 2 {
                                        let rows = t.shape[0] as usize;
                                        let cols = t.shape[1] as usize;
                                        let mut transposed = vec![0.0; rows * cols];
                                        for r in 0..rows {
                                            for c in 0..cols {
                                                transposed[c * rows + r] = t.data[r * cols + c];
                                            }
                                        }
                                        (transposed, vec![t.shape[1], t.shape[0]])
                                    } else {
                                        // For non-2D tensors, just reverse the shape and clone data
                                        let mut rev_shape = t.shape.clone();
                                        rev_shape.reverse();
                                        (t.data.clone(), rev_shape)
                                    }
                                } else {
                                    (vec![], vec![])
                                }
                            };
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(
                                id,
                                TensorState {
                                    data,
                                    shape: new_shape,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_add(a, b) -> handle --
            "rayzor_tensor_add" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                            let (data, shape) = {
                                let s = caller.data();
                                let a = s.tensor_handles.get(&a_h);
                                let b = s.tensor_handles.get(&b_h);
                                match (a, b) {
                                    (Some(a), Some(b)) => {
                                        let len = a.data.len().min(b.data.len());
                                        let data: Vec<f64> =
                                            (0..len).map(|i| a.data[i] + b.data[i]).collect();
                                        (data, a.shape.clone())
                                    }
                                    _ => (vec![], vec![]),
                                }
                            };
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_sub(a, b) -> handle --
            "rayzor_tensor_sub" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                            let (data, shape) = {
                                let s = caller.data();
                                let a = s.tensor_handles.get(&a_h);
                                let b = s.tensor_handles.get(&b_h);
                                match (a, b) {
                                    (Some(a), Some(b)) => {
                                        let len = a.data.len().min(b.data.len());
                                        let data: Vec<f64> =
                                            (0..len).map(|i| a.data[i] - b.data[i]).collect();
                                        (data, a.shape.clone())
                                    }
                                    _ => (vec![], vec![]),
                                }
                            };
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_mul(a, b) -> handle --
            "rayzor_tensor_mul" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                            let (data, shape) = {
                                let s = caller.data();
                                let a = s.tensor_handles.get(&a_h);
                                let b = s.tensor_handles.get(&b_h);
                                match (a, b) {
                                    (Some(a), Some(b)) => {
                                        let len = a.data.len().min(b.data.len());
                                        let data: Vec<f64> =
                                            (0..len).map(|i| a.data[i] * b.data[i]).collect();
                                        (data, a.shape.clone())
                                    }
                                    _ => (vec![], vec![]),
                                }
                            };
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_div(a, b) -> handle --
            "rayzor_tensor_div" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                            let (data, shape) = {
                                let s = caller.data();
                                let a = s.tensor_handles.get(&a_h);
                                let b = s.tensor_handles.get(&b_h);
                                match (a, b) {
                                    (Some(a), Some(b)) => {
                                        let len = a.data.len().min(b.data.len());
                                        let data: Vec<f64> = (0..len)
                                            .map(|i| {
                                                if b.data[i] != 0.0 {
                                                    a.data[i] / b.data[i]
                                                } else {
                                                    f64::NAN
                                                }
                                            })
                                            .collect();
                                        (data, a.shape.clone())
                                    }
                                    _ => (vec![], vec![]),
                                }
                            };
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_matmul(a, b) -> handle (uses wgpu when available) --
            "rayzor_tensor_matmul" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                            let (data, shape) = {
                                let s = caller.data();
                                let a = s.tensor_handles.get(&a_h);
                                let b = s.tensor_handles.get(&b_h);
                                match (a, b) {
                                    (Some(a), Some(b))
                                        if a.shape.len() == 2 && b.shape.len() == 2 =>
                                    {
                                        let m = a.shape[0] as u32;
                                        let k = a.shape[1] as u32;
                                        let n = b.shape[1] as u32;
                                        let result = if k == b.shape[0] as u32 {
                                            if let Some(ref ctx) = s.wgpu_ctx {
                                                // GPU path
                                                ctx.matmul(&a.data, &b.data, m, k, n)
                                            } else {
                                                // CPU fallback
                                                let mut r = vec![0.0; (m * n) as usize];
                                                for i in 0..m as usize {
                                                    for j in 0..n as usize {
                                                        let mut sum = 0.0;
                                                        for p in 0..k as usize {
                                                            sum += a.data[i * k as usize + p]
                                                                * b.data[p * n as usize + j];
                                                        }
                                                        r[i * n as usize + j] = sum;
                                                    }
                                                }
                                                r
                                            }
                                        } else {
                                            vec![]
                                        };
                                        (result, vec![m as i32, n as i32])
                                    }
                                    _ => (vec![], vec![]),
                                }
                            };
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_dot(a, b) -> f64 --
            "rayzor_tensor_dot" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                            let dot = {
                                let s = caller.data();
                                let a = s.tensor_handles.get(&a_h);
                                let b = s.tensor_handles.get(&b_h);
                                match (a, b) {
                                    (Some(a), Some(b)) => {
                                        let len = a.data.len().min(b.data.len());
                                        (0..len).map(|i| a.data[i] * b.data[i]).sum::<f64>()
                                    }
                                    _ => 0.0,
                                }
                            };
                            results[0] = ret_f64(dot, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_sum(handle) -> f64 --
            "rayzor_tensor_sum" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let sum = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| t.data.iter().sum::<f64>())
                                .unwrap_or(0.0);
                            results[0] = ret_f64(sum, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_mean(handle) -> f64 --
            "rayzor_tensor_mean" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let mean = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    if t.data.is_empty() {
                                        0.0
                                    } else {
                                        t.data.iter().sum::<f64>() / t.data.len() as f64
                                    }
                                })
                                .unwrap_or(0.0);
                            results[0] = ret_f64(mean, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_sqrt(handle) -> handle --
            "rayzor_tensor_sqrt" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    (
                                        t.data.iter().map(|x| x.sqrt()).collect::<Vec<_>>(),
                                        t.shape.clone(),
                                    )
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_exp(handle) -> handle --
            "rayzor_tensor_exp" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    (
                                        t.data.iter().map(|x| x.exp()).collect::<Vec<_>>(),
                                        t.shape.clone(),
                                    )
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_log(handle) -> handle --
            "rayzor_tensor_log" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    (
                                        t.data.iter().map(|x| x.ln()).collect::<Vec<_>>(),
                                        t.shape.clone(),
                                    )
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_relu(handle) -> handle --
            "rayzor_tensor_relu" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    (
                                        t.data
                                            .iter()
                                            .map(|&x| if x > 0.0 { x } else { 0.0 })
                                            .collect::<Vec<_>>(),
                                        t.shape.clone(),
                                    )
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_max(handle) -> f64 --
            "rayzor_tensor_max" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let v = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    if t.data.is_empty() {
                                        f64::NEG_INFINITY
                                    } else {
                                        t.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                                    }
                                })
                                .unwrap_or(f64::NEG_INFINITY);
                            results[0] = ret_f64(v, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_min(handle) -> f64 --
            "rayzor_tensor_min" => {
                let rt = ret_ty.clone();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let v = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    if t.data.is_empty() {
                                        f64::INFINITY
                                    } else {
                                        t.data.iter().copied().fold(f64::INFINITY, f64::min)
                                    }
                                })
                                .unwrap_or(f64::INFINITY);
                            results[0] = ret_f64(v, &rt);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_gelu(handle) -> handle --
            // Tanh approximation matching PyTorch `gelu(approximate='tanh')`.
            "rayzor_tensor_gelu" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                                    let out: Vec<f64> = t
                                        .data
                                        .iter()
                                        .map(|&x| {
                                            let inner = c * (x + 0.044715 * x * x * x);
                                            0.5 * x * (1.0 + inner.tanh())
                                        })
                                        .collect();
                                    (out, t.shape.clone())
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_silu(handle) -> handle --
            // x * sigmoid(x), implemented as x / (1 + exp(-x)).
            "rayzor_tensor_silu" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    let out: Vec<f64> =
                                        t.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
                                    (out, t.shape.clone())
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_softmax(handle) -> handle --
            // Softmax over the last dimension (max-subtract for numeric stability).
            "rayzor_tensor_softmax" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    let last = *t.shape.last().unwrap_or(&0).max(&0) as usize;
                                    let total = t.data.len();
                                    if last == 0 || total == 0 {
                                        return (Vec::new(), t.shape.clone());
                                    }
                                    let groups = total / last;
                                    let mut out = vec![0.0_f64; total];
                                    for g in 0..groups {
                                        let base = g * last;
                                        let mut maxv = f64::NEG_INFINITY;
                                        for i in 0..last {
                                            let v = t.data[base + i];
                                            if v > maxv {
                                                maxv = v;
                                            }
                                        }
                                        let mut sum = 0.0;
                                        for i in 0..last {
                                            let e = (t.data[base + i] - maxv).exp();
                                            out[base + i] = e;
                                            sum += e;
                                        }
                                        if sum > 0.0 {
                                            let inv = 1.0 / sum;
                                            for i in 0..last {
                                                out[base + i] *= inv;
                                            }
                                        }
                                    }
                                    (out, t.shape.clone())
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_layer_norm(handle, eps) -> handle --
            // (x - mean(x)) / sqrt(var(x) + eps) over the last dimension.
            "rayzor_tensor_layer_norm" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let eps = val_f64(&params[1]);
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    let last = *t.shape.last().unwrap_or(&0).max(&0) as usize;
                                    let total = t.data.len();
                                    if last == 0 || total == 0 {
                                        return (Vec::new(), t.shape.clone());
                                    }
                                    let groups = total / last;
                                    let n = last as f64;
                                    let mut out = vec![0.0_f64; total];
                                    for g in 0..groups {
                                        let base = g * last;
                                        let mean: f64 =
                                            (0..last).map(|i| t.data[base + i]).sum::<f64>() / n;
                                        let var: f64 = (0..last)
                                            .map(|i| {
                                                let d = t.data[base + i] - mean;
                                                d * d
                                            })
                                            .sum::<f64>()
                                            / n;
                                        let inv = 1.0 / (var + eps).sqrt();
                                        for i in 0..last {
                                            out[base + i] = (t.data[base + i] - mean) * inv;
                                        }
                                    }
                                    (out, t.shape.clone())
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_rms_norm(handle, eps) -> handle --
            // x / sqrt(mean(x^2) + eps) over the last dimension.
            "rayzor_tensor_rms_norm" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let eps = val_f64(&params[1]);
                            let (data, shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    let last = *t.shape.last().unwrap_or(&0).max(&0) as usize;
                                    let total = t.data.len();
                                    if last == 0 || total == 0 {
                                        return (Vec::new(), t.shape.clone());
                                    }
                                    let groups = total / last;
                                    let n = last as f64;
                                    let mut out = vec![0.0_f64; total];
                                    for g in 0..groups {
                                        let base = g * last;
                                        let ms: f64 = (0..last)
                                            .map(|i| {
                                                let v = t.data[base + i];
                                                v * v
                                            })
                                            .sum::<f64>()
                                            / n;
                                        let inv = 1.0 / (ms + eps).sqrt();
                                        for i in 0..last {
                                            out[base + i] = t.data[base + i] * inv;
                                        }
                                    }
                                    (out, t.shape.clone())
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(id, TensorState { data, shape });
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_permute(handle, axesPtr, axesLen) -> handle --
            // Reorders dimensions; the host runtime materializes a new buffer
            // (no stride-based views). axes_ptr points to a contiguous i32
            // array provided by the caller's HaxeArray decomposition.
            "rayzor_tensor_permute" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let axes_ptr = val_i32(&params[1]);
                            let axes_len = val_i32(&params[2]);
                            let axes =
                                read_raw_int_array_i64stride(&mut caller, axes_ptr, axes_len);
                            let axes_len = axes_len as usize;

                            let (data, new_shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    let n = t.shape.len();
                                    if axes.len() != n
                                        || axes.iter().any(|&a| a < 0 || (a as usize) >= n)
                                    {
                                        return (Vec::new(), Vec::new());
                                    }
                                    // Compute row-major strides for original shape
                                    let mut strides = vec![1_usize; n];
                                    for i in (0..n.saturating_sub(1)).rev() {
                                        strides[i] = strides[i + 1] * t.shape[i + 1] as usize;
                                    }
                                    let new_shape: Vec<i32> =
                                        axes.iter().map(|&a| t.shape[a as usize]).collect();
                                    let total: usize =
                                        new_shape.iter().map(|&s| s.max(0) as usize).product();
                                    let new_n = new_shape.len();
                                    let mut new_strides = vec![1_usize; new_n];
                                    for i in (0..new_n.saturating_sub(1)).rev() {
                                        new_strides[i] =
                                            new_strides[i + 1] * new_shape[i + 1] as usize;
                                    }
                                    let mut out = vec![0.0_f64; total];
                                    let mut idx = vec![0_usize; new_n];
                                    for flat in 0..total {
                                        // Decompose flat into multi-index using new_strides
                                        let mut rem = flat;
                                        for (i, s) in new_strides.iter().enumerate() {
                                            idx[i] = rem / s;
                                            rem %= s;
                                        }
                                        // Map back via axes to source offset
                                        let mut src_off = 0_usize;
                                        for (new_dim, &src_dim) in axes.iter().enumerate() {
                                            src_off += idx[new_dim] * strides[src_dim as usize];
                                        }
                                        if src_off < t.data.len() {
                                            out[flat] = t.data[src_off];
                                        }
                                    }
                                    (out, new_shape)
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(
                                id,
                                TensorState {
                                    data,
                                    shape: new_shape,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_slice(handle, dim, start, end) -> handle --
            "rayzor_tensor_slice" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            let dim = val_i32(&params[1]) as usize;
                            let start = val_i32(&params[2]).max(0) as usize;
                            let end = val_i32(&params[3]).max(0) as usize;
                            let (data, new_shape) = caller
                                .data()
                                .tensor_handles
                                .get(&h)
                                .map(|t| {
                                    let n = t.shape.len();
                                    if dim >= n {
                                        return (Vec::new(), Vec::new());
                                    }
                                    let dim_size = t.shape[dim].max(0) as usize;
                                    let e = end.min(dim_size);
                                    if start >= e {
                                        return (Vec::new(), Vec::new());
                                    }
                                    let new_dim = e - start;
                                    let mut new_shape: Vec<i32> = t.shape.clone();
                                    new_shape[dim] = new_dim as i32;

                                    // Row-major strides for original shape
                                    let mut strides = vec![1_usize; n];
                                    for i in (0..n.saturating_sub(1)).rev() {
                                        strides[i] = strides[i + 1] * t.shape[i + 1] as usize;
                                    }
                                    let total: usize =
                                        new_shape.iter().map(|&s| s.max(0) as usize).product();
                                    let mut new_strides = vec![1_usize; n];
                                    for i in (0..n.saturating_sub(1)).rev() {
                                        new_strides[i] =
                                            new_strides[i + 1] * new_shape[i + 1] as usize;
                                    }
                                    let mut out = vec![0.0_f64; total];
                                    let mut idx = vec![0_usize; n];
                                    for flat in 0..total {
                                        let mut rem = flat;
                                        for (i, s) in new_strides.iter().enumerate() {
                                            idx[i] = rem / s;
                                            rem %= s;
                                        }
                                        idx[dim] += start;
                                        let mut src_off = 0_usize;
                                        for (i, st) in strides.iter().enumerate() {
                                            src_off += idx[i] * st;
                                        }
                                        if src_off < t.data.len() {
                                            out[flat] = t.data[src_off];
                                        }
                                    }
                                    (out, new_shape)
                                })
                                .unwrap_or((vec![], vec![]));
                            let s = caller.data_mut();
                            let id = s.next_tensor_id;
                            s.next_tensor_id += 1;
                            s.tensor_handles.insert(
                                id,
                                TensorState {
                                    data,
                                    shape: new_shape,
                                },
                            );
                            results[0] = Val::I32(id);
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_free(handle) -> void --
            "rayzor_tensor_free" => {
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |mut caller, params, results| {
                            let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                            caller.data_mut().tensor_handles.remove(&h);
                            if !results.is_empty() {
                                results[0] = Val::I32(0);
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_data, rayzor_tensor_shape, etc. — stubs returning 0 --
            _ => {
                let results_tys: Vec<ValType> = func_ty.results().collect();
                linker
                    .func_new(
                        "rayzor",
                        name,
                        func_ty.clone(),
                        move |_caller, _params, out| {
                            for (i, r) in results_tys.iter().enumerate() {
                                out[i] = match r {
                                    ValType::I64 => Val::I64(0),
                                    ValType::F32 => Val::F32(0),
                                    ValType::F64 => Val::F64(0),
                                    _ => Val::I32(0),
                                };
                            }
                            Ok(())
                        },
                    )
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }
        }
        registered.insert(name.clone());
    }

    // -- Generic stubs for remaining rayzor imports --
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) {
            continue;
        }
        let results_tys: Vec<ValType> = func_ty.results().collect();
        let name_owned = name.clone();
        linker
            .func_new(
                "rayzor",
                name,
                func_ty.clone(),
                move |_caller, _params, out| {
                    for (i, r) in results_tys.iter().enumerate() {
                        out[i] = match r {
                            ValType::I32 => Val::I32(0),
                            ValType::I64 => Val::I64(0),
                            ValType::F32 => Val::F32(0),
                            ValType::F64 => Val::F64(0),
                            _ => Val::I32(0),
                        };
                    }
                    Ok(())
                },
            )
            .map_err(|e| format!("Failed to stub {}: {}", name_owned, e))?;
    }

    // -- Define shared `env.memory` --
    // The Rayzor runtime-wasm library is built with +atomics and imports
    // its memory from `env.memory`. Allocate a shared memory matching the
    // merged module's declared shape (min=512 pages / 32 MiB so the host-
    // side bump allocator near the top doesn't collide with the runtime
    // heap; max=16384 pages / 1 GiB to match the runtime linker config).
    {
        use wasmtime::{MemoryType, SharedMemory};
        // Bumped from 16384 → 65536 pages (1 GiB → 4 GiB wasm32 cap) so a
        // ~770 MB GGUF can sit in linear memory alongside the runtime-wasm
        // QTensor allocations (each Q6_K weight copies its block buffer).
        let mem_ty = MemoryType::shared(512, 65536);
        let shared = SharedMemory::new(&engine, mem_ty)
            .map_err(|e| format!("Failed to create shared memory: {}", e))?;
        // Clone into WasmState so host functions can read/write it directly:
        // wasmtime doesn't surface imported SharedMemory via
        // `caller.get_export("memory")`, so the fallback path in
        // `read_wasm_mem` / `write_wasm_mem` needs this handle.
        store.data_mut().shared_memory = Some(shared.clone());
        linker
            .define(&mut store, "env", "memory", shared)
            .map_err(|e| format!("Failed to define env.memory: {}", e))?;
    }

    // -- Instantiate & run --
    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|e| format!("WASM instantiation failed: {}", e))?;

    // Initialize host-side bump allocator at top of WASM linear memory.
    // With shared memory imports, `instance.get_memory("memory")` returns
    // None because `env.memory` is imported, not exported. Read the size
    // directly from the stashed SharedMemory instead.
    {
        let exported = instance
            .get_memory(&mut store, "memory")
            .map(|m| m.data_size(&store) as u32);
        let shared_size = store
            .data()
            .shared_memory
            .as_ref()
            .map(|s| s.data_size() as u32);
        // 32 MiB fallback (512 pages × 64 KiB) matches the linker's declared minimum.
        let mem_size = exported.or(shared_size).unwrap_or(512 * 65536);
        // Reserve top 16 bytes for padding; bump allocator grows downward.
        store.data_mut().host_alloc_ptr = mem_size - 16;
    }

    // -- Load the Q8 capability side-module (standard dylink.0 dynamic linking) --
    // If the guest imports the Q8 KV cache and a PIC side-module is available,
    // instantiate it into THIS Store and wire its env/GOT imports per the wasm
    // dynamic-linking ABI, so the Q8 trampolines (registered earlier) dispatch
    // into it. Absent capability/side-module => skip; the guest falls back to
    // the F32 cache via the trampoline's None path. (Phase 1: native, direct
    // .wasm. Eager load — lazy/async is the browser phase.)
    if rayzor_import_names.contains("rayzor_kv_cache_q8_alloc") {
        match find_wasm_side_module() {
            None => eprintln!(
                "[wasm-runner] Q8 imports present but no side-module found; using F32 KV cache"
            ),
            Some(side_path) => {
                let loaded: Result<Instance, String> = (|| {
                    let side_bytes = std::fs::read(&side_path)
                        .map_err(|e| format!("read side module {}: {e}", side_path.display()))?;
                    let side_module = Module::new(&engine, &side_bytes)
                        .map_err(|e| format!("compile side module: {e}"))?;
                    let dylink = parse_side_dylink(&side_bytes)?;

                    // Reserve the side's data image + a private shadow stack from
                    // the runtime heap. rayzor_malloc OWNS it permanently, so the
                    // app heap can't grow into and stomp it (a host_alloc top-carve
                    // could, since rayzor_malloc grows up unbounded). Data size is
                    // the dylink.0 mem_size (16-aligned, per mem_p2align=4) — read
                    // dynamically so it tracks the side-module across rebuilds.
                    let side_mem = (dylink.mem_size + 15) & !15;
                    const SIDE_STACK: u32 = 64 * 1024;
                    let rmalloc = instance
                        .get_typed_func::<i32, i32>(&mut store, "rayzor_malloc")
                        .map_err(|e| format!("main has no rayzor_malloc: {e}"))?;
                    let region = rmalloc
                        .call(&mut store, (side_mem + SIDE_STACK) as i32)
                        .map_err(|e| format!("rayzor_malloc for side region: {e}"))?;
                    if region <= 0 {
                        return Err("rayzor_malloc returned null for side region".into());
                    }
                    let mem_base = region as u32;
                    // Shadow stack lives above the data image; SP starts at the top
                    // and the side's PIC prologue grows it down.
                    let stack_top = mem_base + side_mem + SIDE_STACK;

                    // Base/stack globals — mutability taken from the side's import
                    // declaration (bases Const, SP Var) or Instance::new rejects them.
                    let import_global_mut = |name: &str| -> Mutability {
                        side_module
                            .imports()
                            .find(|i| i.module() == "env" && i.name() == name)
                            .and_then(|i| match i.ty() {
                                ExternType::Global(g) => Some(g.mutability()),
                                _ => None,
                            })
                            .unwrap_or(Mutability::Const)
                    };
                    let mem_base_g = Global::new(
                        &mut store,
                        GlobalType::new(ValType::I32, import_global_mut("__memory_base")),
                        Val::I32(mem_base as i32),
                    )
                    .map_err(|e| format!("__memory_base global: {e}"))?;
                    let table_base_g = Global::new(
                        &mut store,
                        GlobalType::new(ValType::I32, import_global_mut("__table_base")),
                        Val::I32(0),
                    )
                    .map_err(|e| format!("__table_base global: {e}"))?;
                    let sp_g = Global::new(
                        &mut store,
                        GlobalType::new(ValType::I32, import_global_mut("__stack_pointer")),
                        Val::I32(stack_top as i32),
                    )
                    .map_err(|e| format!("__stack_pointer global: {e}"))?;

                    // GOT globals. GOT.func.X = __table_base(0)+slot; GOT.mem.X =
                    // __memory_base+offset. All Var.
                    let mut got_globals: std::collections::HashMap<String, Global> =
                        std::collections::HashMap::new();
                    for (sym, &slot) in &dylink.got_func_slot {
                        let g = Global::new(
                            &mut store,
                            GlobalType::new(ValType::I32, Mutability::Var),
                            Val::I32(slot),
                        )
                        .map_err(|e| format!("GOT.func {sym} global: {e}"))?;
                        got_globals.insert(format!("GOT.func\u{0}{sym}"), g);
                    }
                    for (sym, &off) in &dylink.got_mem_offset {
                        let g = Global::new(
                            &mut store,
                            GlobalType::new(ValType::I32, Mutability::Var),
                            Val::I32(mem_base as i32 + off),
                        )
                        .map_err(|e| format!("GOT.mem {sym} global: {e}"))?;
                        got_globals.insert(format!("GOT.mem\u{0}{sym}"), g);
                    }

                    // Private funcref table (runtime-wasm exports none). Size is
                    // read from the side's import declaration (the elem segment
                    // fills it at __table_base) rather than hard-coded — it tracks
                    // the side-module's GOT/table footprint across rebuilds.
                    let table_min = side_module
                        .imports()
                        .find(|i| i.module() == "env" && i.name() == "__indirect_function_table")
                        .and_then(|i| match i.ty() {
                            ExternType::Table(t) => Some(t.minimum()),
                            _ => None,
                        })
                        .unwrap_or(64);
                    let side_table = Table::new(
                        &mut store,
                        TableType::new(RefType::FUNCREF, table_min as u32, None),
                        Ref::Func(None),
                    )
                    .map_err(|e| format!("side table: {e}"))?;

                    // Pre-resolve main exports the side imports (Copy handles, so the
                    // import map below needs no &mut store).
                    let shared = store
                        .data()
                        .shared_memory
                        .clone()
                        .ok_or("no shared memory for side module")?;
                    let main_malloc = instance
                        .get_func(&mut store, "rayzor_malloc")
                        .ok_or("main missing rayzor_malloc")?;
                    let main_free = instance
                        .get_func(&mut store, "rayzor_free")
                        .ok_or("main missing rayzor_free")?;

                    // Host shims for the libc symbols nothing provides. malloc/calloc
                    // MUST route to MAIN's rayzor_malloc (the shared runtime heap), not
                    // runtime_malloc(), which resolves rayzor_malloc off the *caller's*
                    // exports — and the caller here is the side module, which exports
                    // none. (Critically, LLVM lowers the side's `malloc(n) + memset(0)`
                    // in the Q8 cache alloc to a single `calloc(1, n)` call, so calloc
                    // must allocate from the same heap or the cache handle comes back
                    // null and the guest silently falls back to F32.)
                    let malloc_shim = Func::wrap(
                        &mut store,
                        move |mut caller: Caller<'_, WasmState>, n: i32| -> i32 {
                            main_malloc
                                .typed::<i32, i32>(&caller)
                                .ok()
                                .and_then(|f| f.call(&mut caller, n.max(0)).ok())
                                .unwrap_or(0)
                        },
                    );
                    let calloc = Func::wrap(
                        &mut store,
                        move |mut caller: Caller<'_, WasmState>, nmemb: i32, size: i32| -> i32 {
                            let total = (nmemb as i64 * size as i64).max(0) as i32;
                            let p = main_malloc
                                .typed::<i32, i32>(&caller)
                                .ok()
                                .and_then(|f| f.call(&mut caller, total).ok())
                                .unwrap_or(0);
                            if p != 0 {
                                let zeros = vec![0u8; total as usize];
                                write_wasm_mem(&mut caller, p as u32, &zeros);
                            }
                            p
                        },
                    );
                    let strlen = Func::wrap(
                        &mut store,
                        |mut caller: Caller<'_, WasmState>, p: i32| -> i32 {
                            let mut len = 0u32;
                            while let Some(b) =
                                read_wasm_mem(&mut caller, (p as u32 + len) as usize, 1)
                            {
                                if b[0] == 0 {
                                    break;
                                }
                                len += 1;
                            }
                            len as i32
                        },
                    );
                    let memcmp = Func::wrap(
                        &mut store,
                        |mut caller: Caller<'_, WasmState>, a: i32, b: i32, n: i32| -> i32 {
                            let n = n.max(0) as usize;
                            let av = read_wasm_mem(&mut caller, a as usize, n).unwrap_or_default();
                            let bv = read_wasm_mem(&mut caller, b as usize, n).unwrap_or_default();
                            for i in 0..n.min(av.len()).min(bv.len()) {
                                if av[i] != bv[i] {
                                    return av[i] as i32 - bv[i] as i32;
                                }
                            }
                            0
                        },
                    );
                    let mut tensor_funcs: std::collections::HashMap<String, Func> =
                        std::collections::HashMap::new();
                    for t in [
                        "rayzor_plugin_tensor_dtype",
                        "rayzor_plugin_tensor_is_contiguous",
                        "rayzor_plugin_tensor_ndim",
                        "rayzor_plugin_tensor_shape",
                        "rayzor_plugin_tensor_data",
                        "rayzor_plugin_tensor_alloc_zeros",
                    ] {
                        let f = instance
                            .get_func(&mut store, t)
                            .ok_or_else(|| format!("main missing {t}"))?;
                        tensor_funcs.insert(t.to_string(), f);
                    }

                    // Positional import Vec in the side's DECLARED order (Instance::new
                    // matches by index, not name — wrong order = silent corruption).
                    let mut imports: Vec<Extern> = Vec::new();
                    for imp in side_module.imports() {
                        let ext: Extern = match (imp.module(), imp.name()) {
                            ("env", "memory") => shared.clone().into(),
                            ("env", "__indirect_function_table") => side_table.into(),
                            ("env", "__stack_pointer") => sp_g.into(),
                            ("env", "__memory_base") => mem_base_g.into(),
                            ("env", "__table_base") => table_base_g.into(),
                            ("env", "malloc") => malloc_shim.into(),
                            ("env", "free") => main_free.into(),
                            ("env", "calloc") => calloc.into(),
                            ("env", "strlen") => strlen.into(),
                            ("env", "memcmp") => memcmp.into(),
                            ("env", n) if n.starts_with("rayzor_plugin_tensor_") => tensor_funcs
                                .get(n)
                                .copied()
                                .ok_or_else(|| format!("no tensor fn {n}"))?
                                .into(),
                            ("GOT.func", sym) => (*got_globals
                                .get(&format!("GOT.func\u{0}{sym}"))
                                .ok_or_else(|| format!("unresolved GOT.func {sym}"))?)
                            .into(),
                            ("GOT.mem", sym) => {
                                (*got_globals
                                    .get(&format!("GOT.mem\u{0}{sym}"))
                                    .ok_or_else(|| format!("unresolved GOT.mem {sym}"))?)
                                .into()
                            }
                            (m, n) => return Err(format!("unhandled side import {m}::{n}")),
                        };
                        imports.push(ext);
                    }

                    let side = Instance::new(&mut store, &side_module, &imports)
                        .map_err(|e| format!("instantiate side module: {e}"))?;
                    // Apply data relocations (patches GOT-relative pointers into the
                    // side's .data using the base/GOT globals). No __wasm_call_ctors
                    // in this module.
                    if let Ok(f) =
                        side.get_typed_func::<(), ()>(&mut store, "__wasm_apply_data_relocs")
                    {
                        f.call(&mut store, ())
                            .map_err(|e| format!("__wasm_apply_data_relocs: {e}"))?;
                    }
                    Ok(side)
                })();
                match loaded {
                    Ok(side) => {
                        store.data_mut().side_instance = Some(side);
                        eprintln!(
                            "[wasm-runner] linked Q8 capability side-module ({})",
                            side_path.display()
                        );
                    }
                    Err(e) => eprintln!(
                        "[wasm-runner] WARNING: Q8 side-module load failed ({e}); F32 KV cache fallback"
                    ),
                }
            }
        }
    }

    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .map_err(|e| format!("_start not found: {}", e))?;

    match start.call(&mut store, ()) {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Some(exit) = e.downcast_ref::<wasi_common::I32Exit>() {
                if exit.0 == 0 {
                    return Ok(());
                }
                return Err(format!("process exited with code {}", exit.0));
            }
            Err(format!("WASM execution error: {:?}", e))
        }
    }
}

#[cfg(not(feature = "wasm-runtime"))]
pub fn run_wasm(_wasm_bytes: &[u8]) -> Result<(), String> {
    Err("WASM runtime not available. Install wasmtime or compile rayzor with --features wasm-runtime"
        .to_string())
}
