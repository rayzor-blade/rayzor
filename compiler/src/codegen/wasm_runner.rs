//! WASM Runner — executes compiled WASM binaries via embedded wasmtime.
//!
//! Used by `rayzor run --wasm` to run WASM programs without an external
//! wasmtime installation. Provides WASI P1 imports for stdout, filesystem,
//! environment, and clocks, plus host implementations for haxe.io.Bytes.

#[cfg(feature = "wasm-runtime")]
pub fn run_wasm(wasm_bytes: &[u8]) -> Result<(), String> {
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
        /// Host-side bump allocator: allocates downward from top of WASM memory.
        /// Used to write DynamicValue return structs into WASM linear memory.
        host_alloc_ptr: u32,
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
    fn read_wasm_mem(caller: &mut Caller<'_, WasmState>, addr: usize, len: usize) -> Option<Vec<u8>> {
        let mem = caller.get_export("memory")?.into_memory()?;
        let data = mem.data(&*caller);
        if addr + len <= data.len() {
            Some(data[addr..addr + len].to_vec())
        } else {
            None
        }
    }

    /// Unbox a DynamicValue pointer from WASM memory → i32.
    /// DynamicValue = { type_id: u32, value_ptr: u32 }. Types: 2=Bool, 3=Int.
    fn unbox_int_from_memory(caller: &mut Caller<'_, WasmState>, raw: i32) -> i32 {
        // Heuristic: DynamicValue pointers are heap-allocated (> 64KB) and 4-byte aligned.
        // DynamicValue = { type_id: u32 (0-5), value_ptr: u32 }
        if raw > 65536 && (raw & 3) == 0 {
            if let Some(dv) = read_wasm_mem(caller, raw as usize, 8) {
                let type_id = u32::from_le_bytes(dv[0..4].try_into().unwrap());
                let value_ptr = u32::from_le_bytes(dv[4..8].try_into().unwrap()) as usize;
                // Valid DynamicValue type_ids: 0=Void, 1=Null, 2=Bool, 3=Int, 4=Float, 5=String
                if matches!(type_id, 2 | 3) && value_ptr > 0 && value_ptr < 0x10000000 {
                    if let Some(vb) = read_wasm_mem(caller, value_ptr, 4) {
                        return i32::from_le_bytes(vb[0..4].try_into().unwrap());
                    }
                }
            }
        }
        raw
    }

    fn unbox_f64_from_memory(caller: &mut Caller<'_, WasmState>, raw: i32) -> f64 {
        if raw > 65536 && (raw & 3) == 0 {
            if let Some(dv) = read_wasm_mem(caller, raw as usize, 8) {
                let type_id = u32::from_le_bytes(dv[0..4].try_into().unwrap());
                let value_ptr = u32::from_le_bytes(dv[4..8].try_into().unwrap()) as usize;
                if type_id == 4 && value_ptr > 0 && value_ptr < 0x10000000 {
                    if let Some(vb) = read_wasm_mem(caller, value_ptr, 8) {
                        return f64::from_le_bytes(vb[0..8].try_into().unwrap());
                    }
                }
            }
        }
        raw as f64
    }

    /// Allocate `size` bytes from the host-side bump allocator (top of WASM memory, grows down).
    /// Returns the WASM linear memory address.
    fn host_alloc(caller: &mut Caller<'_, WasmState>, size: u32) -> u32 {
        let ptr = caller.data().host_alloc_ptr;
        let new_ptr = ptr.wrapping_sub(size);
        // Align down to 4 bytes
        let new_ptr = new_ptr & !3;
        caller.data_mut().host_alloc_ptr = new_ptr;
        new_ptr
    }

    /// Write bytes into WASM linear memory at `addr`.
    fn write_wasm_mem(caller: &mut Caller<'_, WasmState>, addr: u32, bytes: &[u8]) {
        if let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory()) {
            let data = mem.data_mut(&mut *caller);
            let a = addr as usize;
            if a + bytes.len() <= data.len() {
                data[a..a + bytes.len()].copy_from_slice(bytes);
            }
        }
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

    /// Read a HaxeString { data_ptr: u32, len: u32 } from WASM memory → Rust String.
    fn read_haxe_string(caller: &mut Caller<'_, WasmState>, str_ptr: i32) -> String {
        let ptr = unbox_int_from_memory(caller, str_ptr) as usize;
        if ptr == 0 { return String::new(); }
        if let Some(header) = read_wasm_mem(caller, ptr, 8) {
            let data_ptr = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            if let Some(bytes) = read_wasm_mem(caller, data_ptr, len) {
                return String::from_utf8_lossy(&bytes).to_string();
            }
        }
        String::new()
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

    /// Read a HaxeArray of i32 values from WASM memory.
    /// HaxeArray layout: { data_ptr: u32, len: u32, cap: u32, elem_size: u32 }.
    fn read_haxe_array_i32(caller: &mut Caller<'_, WasmState>, arr_ptr: i32) -> Vec<i32> {
        let ptr = unbox_int_from_memory(caller, arr_ptr) as usize;
        if ptr == 0 { return vec![]; }
        if let Some(header) = read_wasm_mem(caller, ptr, 16) {
            let data_ptr = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            let elem_size = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
            let actual_size = if elem_size > 0 { elem_size } else { 4 };
            if let Some(data) = read_wasm_mem(caller, data_ptr, len * actual_size) {
                return (0..len).map(|i| {
                    i32::from_le_bytes(data[i*actual_size..i*actual_size+4].try_into().unwrap_or([0;4]))
                }).collect();
            }
        }
        vec![]
    }

    /// Read a HaxeArray of f64 values from WASM memory.
    /// HaxeArray layout: { data_ptr: u32, len: u32, cap: u32, elem_size: u32 }.
    fn read_haxe_array_f64(caller: &mut Caller<'_, WasmState>, arr_ptr: i32) -> Vec<f64> {
        let ptr = unbox_int_from_memory(caller, arr_ptr) as usize;
        if ptr == 0 { return vec![]; }
        // Read HaxeArray header: { data_ptr, len, cap, elem_size }
        let (data_ptr, len, elem_size) = if let Some(header) = read_wasm_mem(caller, ptr, 16) {
            (
                u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize,
                u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize,
                u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize,
            )
        } else {
            return vec![];
        };
        let actual_size = if elem_size > 0 { elem_size } else { 4 };
        if let Some(data) = read_wasm_mem(caller, data_ptr, len * actual_size) {
            // Each element may be a DynamicValue pointer (i32) that wraps an f64
            let raw_vals: Vec<i32> = (0..len).map(|i| {
                i32::from_le_bytes(data[i*actual_size..i*actual_size+4].try_into().unwrap_or([0;4]))
            }).collect();
            // Try to unbox each element as a DynamicValue Float
            let result: Vec<f64> = raw_vals.iter().map(|&raw| {
                unbox_f64_from_memory(caller, raw)
            }).collect();
            return result;
        }
        vec![]
    }

    // -- Engine & module setup --
    let mut config = Config::new();
    config.wasm_simd(true);
    let engine = Engine::new(&config).map_err(|e| format!("Engine config failed: {}", e))?;
    let module =
        Module::new(&engine, wasm_bytes).map_err(|e| format!("WASM compilation failed: {}", e))?;

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
        host_alloc_ptr: 0,
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
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let size = val_i32(&params[0]).max(0) as usize;
                        let id = {
                            let s = caller.data_mut();
                            let id = s.next_bytes_id;
                            s.next_bytes_id += 1;
                            s.bytes_handles.insert(id, vec![0u8; size]);
                            id
                        };
                        results[0] = ret_int(id, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- length(handle) -> i32 --
            "haxe_bytes_length" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let len = caller.data().bytes_handles.get(&h).map(|v| v.len() as i32).unwrap_or(0);
                        results[0] = ret_int(len, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- ofString(str_ptr) -> handle --
            "haxe_bytes_of_string" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let str_ptr = val_i32(&params[0]) as usize;
                        // Read HaxeString { data_ptr: i32, len: i32, cap: i32 } from WASM memory
                        let bytes = {
                            let memory = caller.get_export("memory").and_then(|e| e.into_memory());
                            if let Some(mem) = memory {
                                let data = mem.data(&caller);
                                if str_ptr + 8 <= data.len() {
                                    let data_ptr = u32::from_le_bytes(
                                        data[str_ptr..str_ptr + 4].try_into().unwrap(),
                                    ) as usize;
                                    let len = u32::from_le_bytes(
                                        data[str_ptr + 4..str_ptr + 8].try_into().unwrap(),
                                    ) as usize;
                                    if data_ptr + len <= data.len() {
                                        data[data_ptr..data_ptr + len].to_vec()
                                    } else {
                                        vec![]
                                    }
                                } else {
                                    vec![]
                                }
                            } else {
                                vec![]
                            }
                        };
                        let id = {
                            let s = caller.data_mut();
                            let id = s.next_bytes_id;
                            s.next_bytes_id += 1;
                            s.bytes_handles.insert(id, bytes);
                            id
                        };
                        results[0] = ret_int(id, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- get(handle, pos) -> byte --
            "haxe_bytes_get" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller
                            .data()
                            .bytes_handles
                            .get(&h)
                            .and_then(|v| v.get(pos))
                            .copied()
                            .unwrap_or(0) as i32;
                        let boxed = box_int_in_wasm(&mut caller, val);
                        results[0] = ret_int(boxed, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- set(handle, pos, val) --
            "haxe_bytes_set" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let raw_val = val_i32(&params[2]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = unbox_int_from_memory(&mut caller, raw_val) as u8;
                        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&h) {
                            if pos < v.len() {
                                v[pos] = val;
                            }
                        }
                        if !results.is_empty() { results[0] = Val::I32(0); }
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getInt16(handle, pos) -> i32 --
            "haxe_bytes_get_int16" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 2 <= v.len() {
                                i16::from_le_bytes(v[pos..pos + 2].try_into().unwrap()) as i32
                            } else {
                                0
                            }
                        }).unwrap_or(0);
                        let boxed = box_int_in_wasm(&mut caller, val);
                        results[0] = ret_int(boxed, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt16(handle, pos, val) --
            "haxe_bytes_set_int16" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let raw_val = val_i32(&params[2]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = unbox_int_from_memory(&mut caller, raw_val) as i16;
                        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&h) {
                            if pos + 2 <= v.len() {
                                v[pos..pos + 2].copy_from_slice(&val.to_le_bytes());
                            }
                        }
                        if !results.is_empty() { results[0] = Val::I32(0); }
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getInt32(handle, pos) -> i32 --
            "haxe_bytes_get_int32" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 4 <= v.len() {
                                i32::from_le_bytes(v[pos..pos + 4].try_into().unwrap())
                            } else {
                                0
                            }
                        }).unwrap_or(0);
                        let boxed = box_int_in_wasm(&mut caller, val);
                        results[0] = ret_int(boxed, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt32(handle, pos, val) --
            "haxe_bytes_set_int32" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let raw_val = val_i32(&params[2]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = unbox_int_from_memory(&mut caller, raw_val);
                        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&h) {
                            if pos + 4 <= v.len() {
                                v[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
                            }
                        }
                        if !results.is_empty() { results[0] = Val::I32(0); }
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getInt64(handle, pos) -> i64 --
            "haxe_bytes_get_int64" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 8 <= v.len() {
                                i64::from_le_bytes(v[pos..pos + 8].try_into().unwrap())
                            } else {
                                0
                            }
                        }).unwrap_or(0);
                        let boxed = box_int_in_wasm(&mut caller, val as i32);
                        results[0] = ret_int(boxed, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt64(handle, pos, val) --
            "haxe_bytes_set_int64" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let raw_val = val_i32(&params[2]);
                        let val = unbox_int_from_memory(&mut caller, raw_val) as i64;
                        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&h) {
                            if pos + 8 <= v.len() {
                                v[pos..pos + 8].copy_from_slice(&val.to_le_bytes());
                            }
                        }
                        if !results.is_empty() { results[0] = Val::I32(0); }
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getFloat(handle, pos) -> f32 (returned as Haxe Float = f64) --
            "haxe_bytes_get_float" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 4 <= v.len() {
                                f32::from_le_bytes(v[pos..pos + 4].try_into().unwrap())
                            } else {
                                0.0
                            }
                        }).unwrap_or(0.0);
                        let boxed = box_float_in_wasm(&mut caller, val as f64);
                        results[0] = ret_int(boxed, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setFloat(handle, pos, val) --
            "haxe_bytes_set_float" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let raw_val = val_i32(&params[2]);
                        let val = unbox_f64_from_memory(&mut caller, raw_val) as f32;
                        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&h) {
                            if pos + 4 <= v.len() {
                                v[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
                            }
                        }
                        if !results.is_empty() { results[0] = Val::I32(0); }
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- getDouble(handle, pos) -> f64 --
            "haxe_bytes_get_double" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 8 <= v.len() {
                                f64::from_le_bytes(v[pos..pos + 8].try_into().unwrap())
                            } else {
                                0.0
                            }
                        }).unwrap_or(0.0);
                        let boxed = box_float_in_wasm(&mut caller, val);
                        results[0] = ret_int(boxed, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setDouble(handle, pos, val) --
            "haxe_bytes_set_double" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let raw_val = val_i32(&params[2]);
                        let val = unbox_f64_from_memory(&mut caller, raw_val);
                        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&h) {
                            if pos + 8 <= v.len() {
                                v[pos..pos + 8].copy_from_slice(&val.to_le_bytes());
                            }
                        }
                        if !results.is_empty() { results[0] = Val::I32(0); }
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- fill(handle, pos, len, val) --
            "haxe_bytes_fill" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let raw_len = val_i32(&params[2]);
                        let raw_val = val_i32(&params[3]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let len = unbox_int_from_memory(&mut caller, raw_len) as usize;
                        let val = unbox_int_from_memory(&mut caller, raw_val) as u8;
                        if let Some(v) = caller.data_mut().bytes_handles.get_mut(&h) {
                            let end = (pos + len).min(v.len());
                            if pos < end {
                                v[pos..end].fill(val);
                            }
                        }
                        if !results.is_empty() { results[0] = Val::I32(0); }
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- blit(dest, destPos, src, srcPos, len) --
            "haxe_bytes_blit" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let dest_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_dest_pos = val_i32(&params[1]);
                        let src_h = unbox_int_from_memory(&mut caller, val_i32(&params[2]));
                        let raw_src_pos = val_i32(&params[3]);
                        let raw_len = val_i32(&params[4]);
                        let dest_pos = unbox_int_from_memory(&mut caller, raw_dest_pos) as usize;
                        let src_pos = unbox_int_from_memory(&mut caller, raw_src_pos) as usize;
                        let len = unbox_int_from_memory(&mut caller, raw_len) as usize;
                        // Copy src bytes first (to handle overlapping handles)
                        let src_bytes: Vec<u8> = caller
                            .data()
                            .bytes_handles
                            .get(&src_h)
                            .map(|v| {
                                let end = (src_pos + len).min(v.len());
                                if src_pos < end {
                                    v[src_pos..end].to_vec()
                                } else {
                                    vec![]
                                }
                            })
                            .unwrap_or_default();
                        if let Some(dest) = caller.data_mut().bytes_handles.get_mut(&dest_h) {
                            let copy_len = src_bytes.len().min(dest.len().saturating_sub(dest_pos));
                            if copy_len > 0 {
                                dest[dest_pos..dest_pos + copy_len]
                                    .copy_from_slice(&src_bytes[..copy_len]);
                            }
                        }
                        if !results.is_empty() { results[0] = Val::I32(0); }
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- compare(a, b) -> i32 --
            "haxe_bytes_compare" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                        let cmp = {
                            let s = caller.data();
                            let a = s.bytes_handles.get(&a_h).map(|v| v.as_slice()).unwrap_or(&[]);
                            let b = s.bytes_handles.get(&b_h).map(|v| v.as_slice()).unwrap_or(&[]);
                            a.cmp(b) as i32
                        };
                        let boxed = box_int_in_wasm(&mut caller, cmp);
                        results[0] = ret_int(boxed, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- sub(handle, pos, len) -> handle --
            "haxe_bytes_sub" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                        let raw_pos = val_i32(&params[1]);
                        let raw_len = val_i32(&params[2]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let len = unbox_int_from_memory(&mut caller, raw_len) as usize;
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
                        let id = {
                            let s = caller.data_mut();
                            let id = s.next_bytes_id;
                            s.next_bytes_id += 1;
                            s.bytes_handles.insert(id, sub);
                            id
                        };
                        results[0] = ret_int(id, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            _ => continue,
        }

        registered.insert(name.clone());
    }

    // -- Register EReg host functions --
    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) { continue; }
        let is_ereg = matches!(name.as_str(),
            "haxe_ereg_new" | "haxe_ereg_match" | "haxe_ereg_matched"
            | "haxe_ereg_matched_left" | "haxe_ereg_matched_right"
            | "haxe_ereg_matched_pos" | "haxe_ereg_matched_pos_anon"
            | "haxe_ereg_match_sub" | "haxe_ereg_replace" | "haxe_ereg_escape"
            | "haxe_ereg_split" | "haxe_ereg_map"
        );
        if !is_ereg { continue; }

        match name.as_str() {
            // new(pattern, flags) -> handle
            "haxe_ereg_new" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let pattern = read_haxe_string(&mut caller, val_i32(&params[0]));
                    let flags = read_haxe_string(&mut caller, val_i32(&params[1]));
                    let mut re_pattern = pattern.clone();
                    // Convert Haxe regex flags to Rust regex flags
                    let case_insensitive = flags.contains('i');
                    let multiline = flags.contains('m');
                    let dotall = flags.contains('s');
                    if case_insensitive || multiline || dotall {
                        let mut prefix = String::from("(?");
                        if case_insensitive { prefix.push('i'); }
                        if multiline { prefix.push('m'); }
                        if dotall { prefix.push('s'); }
                        prefix.push(')');
                        re_pattern = format!("{}{}", prefix, re_pattern);
                    }
                    let regex = match regex::Regex::new(&re_pattern) {
                        Ok(r) => r,
                        Err(_) => { results[0] = Val::I32(0); return Ok(()); }
                    };
                    let s = caller.data_mut();
                    let id = s.next_ereg_id;
                    s.next_ereg_id += 1;
                    s.ereg_handles.insert(id, ERegState {
                        pattern, flags, regex,
                        last_input: None, last_match: None, last_captures: vec![],
                        match_start: 0, match_end: 0,
                    });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // match(this, s) -> bool (boxed)
            "haxe_ereg_match" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let s = read_haxe_string(&mut caller, val_i32(&params[1]));
                    let matched = {
                        if let Some(st) = caller.data_mut().ereg_handles.get_mut(&h) {
                            if let Some(caps) = st.regex.captures(&s) {
                                st.match_start = caps.get(0).map(|m| m.start()).unwrap_or(0);
                                st.match_end = caps.get(0).map(|m| m.end()).unwrap_or(0);
                                st.last_captures = (0..caps.len()).map(|i| caps.get(i).map(|m| m.as_str().to_string())).collect();
                                st.last_input = Some(s);
                                true
                            } else {
                                st.last_input = Some(s);
                                st.last_captures.clear();
                                false
                            }
                        } else { false }
                    };
                    let boxed = box_int_in_wasm(&mut caller, if matched { 1 } else { 0 });
                    results[0] = Val::I32(boxed);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // matched(this, n) -> String
            "haxe_ereg_matched" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let n = unbox_int_from_memory(&mut caller, val_i32(&params[1])) as usize;
                    let val = caller.data().ereg_handles.get(&h)
                        .and_then(|st| st.last_captures.get(n).cloned().flatten())
                        .unwrap_or_default();
                    let ptr = write_haxe_string(&mut caller, &val);
                    results[0] = Val::I32(ptr);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // matchedLeft(this) -> String
            "haxe_ereg_matched_left" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let val = caller.data().ereg_handles.get(&h)
                        .and_then(|st| st.last_input.as_ref().map(|s| s[..st.match_start].to_string()))
                        .unwrap_or_default();
                    let ptr = write_haxe_string(&mut caller, &val);
                    results[0] = Val::I32(ptr);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // matchedRight(this) -> String
            "haxe_ereg_matched_right" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let val = caller.data().ereg_handles.get(&h)
                        .and_then(|st| st.last_input.as_ref().map(|s| s[st.match_end..].to_string()))
                        .unwrap_or_default();
                    let ptr = write_haxe_string(&mut caller, &val);
                    results[0] = Val::I32(ptr);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // replace(this, s, by) -> String
            "haxe_ereg_replace" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let s = read_haxe_string(&mut caller, val_i32(&params[1]));
                    let by = read_haxe_string(&mut caller, val_i32(&params[2]));
                    let replaced = caller.data().ereg_handles.get(&h)
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
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // static escape(s) -> String
            "haxe_ereg_escape" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let s = read_haxe_string(&mut caller, val_i32(&params[0]));
                    let escaped = regex::escape(&s);
                    let ptr = write_haxe_string(&mut caller, &escaped);
                    results[0] = Val::I32(ptr);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // matchSub, matchedPos, split, map — return stubs for now
            _ => {
                let results_tys: Vec<ValType> = func_ty.results().collect();
                linker.func_new("rayzor", name, func_ty.clone(), move |_caller, _params, out| {
                    for (i, r) in results_tys.iter().enumerate() {
                        out[i] = match r {
                            ValType::I64 => Val::I64(0),
                            ValType::F32 => Val::F32(0),
                            ValType::F64 => Val::F64(0),
                            _ => Val::I32(0),
                        };
                    }
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }
        }
        registered.insert(name.clone());
    }

    // -- Register Mutex/Arc/Box host functions --
    fn canonical_sync_name(name: &str) -> Option<&str> {
        match name {
            // Qualified names
            "rayzor_mutex_init" | "rayzor_mutex_lock" | "rayzor_mutex_try_lock"
            | "rayzor_mutex_is_locked" | "rayzor_mutex_unlock" | "rayzor_mutex_guard_get"
            | "mutex_guard_unlock" | "MutexGuard_unlock"
            | "rayzor_arc_init" | "rayzor_arc_clone" | "rayzor_arc_get"
            | "rayzor_arc_as_ptr" | "rayzor_arc_try_unwrap" | "rayzor_arc_strong_count"
            | "rayzor_box_init" | "rayzor_box_unbox" | "rayzor_box_raw" | "rayzor_box_free"
                => Some(name),
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
        if registered.contains(name) { continue; }
        let canon = match canonical_sync_name(name) {
            Some(c) => c,
            None => continue,
        };

        match canon {
            // -- rayzor_mutex_init(val) -> handle (raw, NOT boxed) --
            "rayzor_mutex_init" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let val = val_i32(&params[0]);
                    let s = caller.data_mut();
                    let id = s.next_mutex_id;
                    s.next_mutex_id += 1;
                    s.mutex_handles.insert(id, MutexState { locked: false, value: val });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_lock(handle) -> boxed guard handle --
            "rayzor_mutex_lock" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    if let Some(st) = caller.data_mut().mutex_handles.get_mut(&h) {
                        st.locked = true;
                    }
                    let boxed = box_int_in_wasm(&mut caller, h);
                    results[0] = Val::I32(boxed);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_try_lock(handle) -> boxed 1 if acquired, boxed 0 if already locked --
            "rayzor_mutex_try_lock" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
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
                    let boxed = box_bool_in_wasm(&mut caller, acquired);
                    results[0] = Val::I32(boxed);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_is_locked(handle) -> boxed 1 if locked, boxed 0 if not --
            "rayzor_mutex_is_locked" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let locked = caller.data().mutex_handles.get(&h)
                        .map(|st| if st.locked { 1 } else { 0 })
                        .unwrap_or(0);
                    let boxed = box_bool_in_wasm(&mut caller, locked);
                    results[0] = Val::I32(boxed);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_unlock(handle) -> void --
            "rayzor_mutex_unlock" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    if let Some(st) = caller.data_mut().mutex_handles.get_mut(&h) {
                        st.locked = false;
                    }
                    if !results.is_empty() { results[0] = Val::I32(0); }
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_mutex_guard_get(handle) -> boxed value --
            "rayzor_mutex_guard_get" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let val = caller.data().mutex_handles.get(&h)
                        .map(|st| st.value)
                        .unwrap_or(0);
                    let boxed = box_int_in_wasm(&mut caller, val);
                    results[0] = Val::I32(boxed);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- Arc: identity pass-through --
            "rayzor_arc_init" | "rayzor_arc_clone" | "rayzor_arc_get"
            | "rayzor_arc_as_ptr" | "rayzor_arc_try_unwrap" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |_caller, params, results| {
                    results[0] = params[0].clone();
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_arc_strong_count -> boxed 1 --
            "rayzor_arc_strong_count" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, _params, results| {
                    let boxed = box_int_in_wasm(&mut caller, 1);
                    results[0] = Val::I32(boxed);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- Box: identity pass-through --
            "rayzor_box_init" | "rayzor_box_unbox" | "rayzor_box_raw" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |_caller, params, results| {
                    results[0] = params[0].clone();
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_box_free -> no-op --
            "rayzor_box_free" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |_caller, _params, results| {
                    if !results.is_empty() { results[0] = Val::I32(0); }
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
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
            "rayzor_tensor_from_array" | "Tensor_fromArray" | "Tensor_from_array" => Some("rayzor_tensor_from_array"),
            "rayzor_tensor_rand" | "Tensor_rand" => Some("rayzor_tensor_rand"),
            "rayzor_tensor_ndim" => Some("rayzor_tensor_ndim"),
            "rayzor_tensor_numel" => Some("rayzor_tensor_numel"),
            "rayzor_tensor_dtype" => Some("rayzor_tensor_dtype"),
            "rayzor_tensor_get" => Some("rayzor_tensor_get"),
            "rayzor_tensor_set" => Some("rayzor_tensor_set"),
            "rayzor_tensor_reshape" => Some("rayzor_tensor_reshape"),
            "rayzor_tensor_transpose" => Some("rayzor_tensor_transpose"),
            "rayzor_tensor_add" | "Tensor_add" => Some("rayzor_tensor_add"),
            "rayzor_tensor_sub" | "Tensor_sub" => Some("rayzor_tensor_sub"),
            "rayzor_tensor_mul" | "Tensor_mul" => Some("rayzor_tensor_mul"),
            "rayzor_tensor_div" | "Tensor_div" => Some("rayzor_tensor_div"),
            "rayzor_tensor_matmul" => Some("rayzor_tensor_matmul"),
            "rayzor_tensor_dot" => Some("rayzor_tensor_dot"),
            "rayzor_tensor_sum" => Some("rayzor_tensor_sum"),
            "rayzor_tensor_mean" => Some("rayzor_tensor_mean"),
            "rayzor_tensor_sqrt" => Some("rayzor_tensor_sqrt"),
            "rayzor_tensor_exp" => Some("rayzor_tensor_exp"),
            "rayzor_tensor_log" => Some("rayzor_tensor_log"),
            "rayzor_tensor_relu" => Some("rayzor_tensor_relu"),
            "rayzor_tensor_free" => Some("rayzor_tensor_free"),
            "rayzor_tensor_data" | "rayzor_tensor_shape" | "rayzor_tensor_shape_ptr"
            | "rayzor_tensor_shape_ndim" => Some(name),
            _ => None,
        }
    }

    for (name, func_ty) in &rayzor_imports {
        if registered.contains(name) { continue; }
        let canon = match canonical_tensor_name(name) {
            Some(c) => c,
            None => continue,
        };
        let ret_ty: ValType = func_ty.results().next().unwrap_or(ValType::I32);

        match canon {
            // -- rayzor_tensor_zeros(shapePtr, dtype) -> handle --
            "rayzor_tensor_zeros" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let shape_ptr = val_i32(&params[0]);
                    let shape = read_haxe_array_i32(&mut caller, shape_ptr);
                    let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data: vec![0.0; numel], shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_ones(shapePtr, dtype) -> handle --
            "rayzor_tensor_ones" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let shape_ptr = val_i32(&params[0]);
                    let shape = read_haxe_array_i32(&mut caller, shape_ptr);
                    let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data: vec![1.0; numel], shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_full(shapePtr, value, dtype) -> handle --
            "rayzor_tensor_full" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let shape_ptr = val_i32(&params[0]);
                    let value = val_f64(&params[1]);
                    let shape = read_haxe_array_i32(&mut caller, shape_ptr);
                    let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data: vec![value; numel], shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_from_array(dataPtr, dtype) -> handle --
            "rayzor_tensor_from_array" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let data_ptr = val_i32(&params[0]);
                    let data = read_haxe_array_f64(&mut caller, data_ptr);
                    let len = data.len() as i32;
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data, shape: vec![len] });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_rand(shapePtr, dtype) -> handle --
            "rayzor_tensor_rand" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let shape_ptr = val_i32(&params[0]);
                    let shape = read_haxe_array_i32(&mut caller, shape_ptr);
                    let numel: usize = shape.iter().map(|&s| s.max(0) as usize).product();
                    // Simple LCG pseudo-random for determinism in WASM
                    let mut seed: u64 = 12345;
                    let data: Vec<f64> = (0..numel).map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as f64 / (1u64 << 31) as f64
                    }).collect();
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data, shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_ndim(handle) -> raw int --
            "rayzor_tensor_ndim" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let ndim = caller.data().tensor_handles.get(&h)
                        .map(|t| t.shape.len() as i32)
                        .unwrap_or(0);
                    results[0] = Val::I32(ndim);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_numel(handle) -> int --
            "rayzor_tensor_numel" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let numel = caller.data().tensor_handles.get(&h)
                        .map(|t| t.data.len() as i32)
                        .unwrap_or(0);
                    results[0] = Val::I32(numel);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_dtype(handle) -> raw int (0 = Float64) --
            "rayzor_tensor_dtype" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, _params, results| {
                    results[0] = Val::I32(0); // Float64 = 0
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_get(handle, idx) -> f64 --
            "rayzor_tensor_get" => {
                let rt = ret_ty.clone();
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let idx = unbox_int_from_memory(&mut caller, val_i32(&params[1])) as usize;
                    let val = caller.data().tensor_handles.get(&h)
                        .and_then(|t| t.data.get(idx).copied())
                        .unwrap_or(0.0);
                    results[0] = ret_f64(val, &rt);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_set(handle, idx, value) -> void --
            "rayzor_tensor_set" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let idx = unbox_int_from_memory(&mut caller, val_i32(&params[1])) as usize;
                    let value = val_f64(&params[2]);
                    if let Some(t) = caller.data_mut().tensor_handles.get_mut(&h) {
                        if idx < t.data.len() {
                            t.data[idx] = value;
                        }
                    }
                    if !results.is_empty() { results[0] = Val::I32(0); }
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_reshape(handle, shapePtr) -> handle --
            "rayzor_tensor_reshape" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let shape_ptr = val_i32(&params[1]);
                    let new_shape = read_haxe_array_i32(&mut caller, shape_ptr);
                    let data = caller.data().tensor_handles.get(&h)
                        .map(|t| t.data.clone())
                        .unwrap_or_default();
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data, shape: new_shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_transpose(handle) -> handle --
            "rayzor_tensor_transpose" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
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
                    s.tensor_handles.insert(id, TensorState { data, shape: new_shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_add(a, b) -> handle --
            "rayzor_tensor_add" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                    let (data, shape) = {
                        let s = caller.data();
                        let a = s.tensor_handles.get(&a_h);
                        let b = s.tensor_handles.get(&b_h);
                        match (a, b) {
                            (Some(a), Some(b)) => {
                                let len = a.data.len().min(b.data.len());
                                let data: Vec<f64> = (0..len).map(|i| a.data[i] + b.data[i]).collect();
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
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_sub(a, b) -> handle --
            "rayzor_tensor_sub" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                    let (data, shape) = {
                        let s = caller.data();
                        let a = s.tensor_handles.get(&a_h);
                        let b = s.tensor_handles.get(&b_h);
                        match (a, b) {
                            (Some(a), Some(b)) => {
                                let len = a.data.len().min(b.data.len());
                                let data: Vec<f64> = (0..len).map(|i| a.data[i] - b.data[i]).collect();
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
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_mul(a, b) -> handle --
            "rayzor_tensor_mul" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                    let (data, shape) = {
                        let s = caller.data();
                        let a = s.tensor_handles.get(&a_h);
                        let b = s.tensor_handles.get(&b_h);
                        match (a, b) {
                            (Some(a), Some(b)) => {
                                let len = a.data.len().min(b.data.len());
                                let data: Vec<f64> = (0..len).map(|i| a.data[i] * b.data[i]).collect();
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
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_div(a, b) -> handle --
            "rayzor_tensor_div" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                    let (data, shape) = {
                        let s = caller.data();
                        let a = s.tensor_handles.get(&a_h);
                        let b = s.tensor_handles.get(&b_h);
                        match (a, b) {
                            (Some(a), Some(b)) => {
                                let len = a.data.len().min(b.data.len());
                                let data: Vec<f64> = (0..len).map(|i| {
                                    if b.data[i] != 0.0 { a.data[i] / b.data[i] } else { f64::NAN }
                                }).collect();
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
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_matmul(a, b) -> handle --
            "rayzor_tensor_matmul" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let a_h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let b_h = unbox_int_from_memory(&mut caller, val_i32(&params[1]));
                    let (data, shape) = {
                        let s = caller.data();
                        let a = s.tensor_handles.get(&a_h);
                        let b = s.tensor_handles.get(&b_h);
                        match (a, b) {
                            (Some(a), Some(b)) if a.shape.len() == 2 && b.shape.len() == 2 => {
                                let m = a.shape[0] as usize;
                                let k = a.shape[1] as usize;
                                let n = b.shape[1] as usize;
                                let mut result = vec![0.0; m * n];
                                if k == b.shape[0] as usize {
                                    for i in 0..m {
                                        for j in 0..n {
                                            let mut sum = 0.0;
                                            for p in 0..k {
                                                sum += a.data[i * k + p] * b.data[p * n + j];
                                            }
                                            result[i * n + j] = sum;
                                        }
                                    }
                                }
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
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_dot(a, b) -> f64 --
            "rayzor_tensor_dot" => {
                let rt = ret_ty.clone();
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
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
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_sum(handle) -> f64 --
            "rayzor_tensor_sum" => {
                let rt = ret_ty.clone();
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let sum = caller.data().tensor_handles.get(&h)
                        .map(|t| t.data.iter().sum::<f64>())
                        .unwrap_or(0.0);
                    results[0] = ret_f64(sum, &rt);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_mean(handle) -> f64 --
            "rayzor_tensor_mean" => {
                let rt = ret_ty.clone();
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let mean = caller.data().tensor_handles.get(&h)
                        .map(|t| {
                            if t.data.is_empty() { 0.0 }
                            else { t.data.iter().sum::<f64>() / t.data.len() as f64 }
                        })
                        .unwrap_or(0.0);
                    results[0] = ret_f64(mean, &rt);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_sqrt(handle) -> handle --
            "rayzor_tensor_sqrt" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let (data, shape) = caller.data().tensor_handles.get(&h)
                        .map(|t| (t.data.iter().map(|x| x.sqrt()).collect::<Vec<_>>(), t.shape.clone()))
                        .unwrap_or((vec![], vec![]));
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data, shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_exp(handle) -> handle --
            "rayzor_tensor_exp" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let (data, shape) = caller.data().tensor_handles.get(&h)
                        .map(|t| (t.data.iter().map(|x| x.exp()).collect::<Vec<_>>(), t.shape.clone()))
                        .unwrap_or((vec![], vec![]));
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data, shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_log(handle) -> handle --
            "rayzor_tensor_log" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let (data, shape) = caller.data().tensor_handles.get(&h)
                        .map(|t| (t.data.iter().map(|x| x.ln()).collect::<Vec<_>>(), t.shape.clone()))
                        .unwrap_or((vec![], vec![]));
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data, shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_relu(handle) -> handle --
            "rayzor_tensor_relu" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    let (data, shape) = caller.data().tensor_handles.get(&h)
                        .map(|t| (t.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect::<Vec<_>>(), t.shape.clone()))
                        .unwrap_or((vec![], vec![]));
                    let s = caller.data_mut();
                    let id = s.next_tensor_id;
                    s.next_tensor_id += 1;
                    s.tensor_handles.insert(id, TensorState { data, shape });
                    results[0] = Val::I32(id);
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_free(handle) -> void --
            "rayzor_tensor_free" => {
                linker.func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                    let h = unbox_int_from_memory(&mut caller, val_i32(&params[0]));
                    caller.data_mut().tensor_handles.remove(&h);
                    if !results.is_empty() { results[0] = Val::I32(0); }
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- rayzor_tensor_data, rayzor_tensor_shape, etc. — stubs returning 0 --
            _ => {
                let results_tys: Vec<ValType> = func_ty.results().collect();
                linker.func_new("rayzor", name, func_ty.clone(), move |_caller, _params, out| {
                    for (i, r) in results_tys.iter().enumerate() {
                        out[i] = match r {
                            ValType::I64 => Val::I64(0),
                            ValType::F32 => Val::F32(0),
                            ValType::F64 => Val::F64(0),
                            _ => Val::I32(0),
                        };
                    }
                    Ok(())
                }).map_err(|e| format!("Failed to register {}: {}", name, e))?;
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

    // -- Instantiate & run --
    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|e| format!("WASM instantiation failed: {}", e))?;

    // Initialize host-side bump allocator at top of WASM linear memory
    if let Some(mem) = instance.get_memory(&mut store, "memory") {
        let mem_size = mem.data_size(&store) as u32;
        // Reserve top 256KB for host allocations (boxed return values)
        store.data_mut().host_alloc_ptr = mem_size - 16; // start near top, 16-byte aligned
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
