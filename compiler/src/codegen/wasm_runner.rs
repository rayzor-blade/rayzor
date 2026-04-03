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
        next_bytes_id: 1, // 0 = null handle
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
                        let s = caller.data_mut();
                        let id = s.next_bytes_id;
                        s.next_bytes_id += 1;
                        s.bytes_handles.insert(id, vec![0u8; size]);
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
                        let h = val_i32(&params[0]);
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
                        let s = caller.data_mut();
                        let id = s.next_bytes_id;
                        s.next_bytes_id += 1;
                        s.bytes_handles.insert(id, bytes);
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
                        let h = val_i32(&params[0]);
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller
                            .data()
                            .bytes_handles
                            .get(&h)
                            .and_then(|v| v.get(pos))
                            .copied()
                            .unwrap_or(0) as i32;
                        results[0] = ret_int(val, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- set(handle, pos, val) --
            "haxe_bytes_set" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = val_i32(&params[0]);
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
                        let h = val_i32(&params[0]);
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 2 <= v.len() {
                                i16::from_le_bytes(v[pos..pos + 2].try_into().unwrap()) as i32
                            } else {
                                0
                            }
                        }).unwrap_or(0);
                        results[0] = ret_int(val, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt16(handle, pos, val) --
            "haxe_bytes_set_int16" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = val_i32(&params[0]);
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
                        let h = val_i32(&params[0]);
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 4 <= v.len() {
                                i32::from_le_bytes(v[pos..pos + 4].try_into().unwrap())
                            } else {
                                0
                            }
                        }).unwrap_or(0);
                        results[0] = ret_int(val, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt32(handle, pos, val) --
            "haxe_bytes_set_int32" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = val_i32(&params[0]);
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
                        let h = val_i32(&params[0]);
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 8 <= v.len() {
                                i64::from_le_bytes(v[pos..pos + 8].try_into().unwrap())
                            } else {
                                0
                            }
                        }).unwrap_or(0);
                        results[0] = match rt {
                            ValType::I64 => Val::I64(val),
                            ValType::I32 => Val::I32(val as i32),
                            _ => Val::I64(val),
                        };
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setInt64(handle, pos, val) --
            "haxe_bytes_set_int64" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = val_i32(&params[0]);
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
                        let h = val_i32(&params[0]);
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 4 <= v.len() {
                                f32::from_le_bytes(v[pos..pos + 4].try_into().unwrap())
                            } else {
                                0.0
                            }
                        }).unwrap_or(0.0);
                        results[0] = ret_f32(val, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setFloat(handle, pos, val) --
            "haxe_bytes_set_float" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = val_i32(&params[0]);
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
                        let h = val_i32(&params[0]);
                        let raw_pos = val_i32(&params[1]);
                        let pos = unbox_int_from_memory(&mut caller, raw_pos) as usize;
                        let val = caller.data().bytes_handles.get(&h).map(|v| {
                            if pos + 8 <= v.len() {
                                f64::from_le_bytes(v[pos..pos + 8].try_into().unwrap())
                            } else {
                                0.0
                            }
                        }).unwrap_or(0.0);
                        results[0] = ret_f64(val, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- setDouble(handle, pos, val) --
            "haxe_bytes_set_double" => {
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = val_i32(&params[0]);
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
                        let h = val_i32(&params[0]);
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
                        let dest_h = val_i32(&params[0]);
                        let raw_dest_pos = val_i32(&params[1]);
                        let src_h = val_i32(&params[2]);
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
                        let a_h = val_i32(&params[0]);
                        let b_h = val_i32(&params[1]);
                        let s = caller.data();
                        let a = s.bytes_handles.get(&a_h).map(|v| v.as_slice()).unwrap_or(&[]);
                        let b = s.bytes_handles.get(&b_h).map(|v| v.as_slice()).unwrap_or(&[]);
                        let cmp = a.cmp(b) as i32;
                        results[0] = ret_int(cmp, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            // -- sub(handle, pos, len) -> handle --
            "haxe_bytes_sub" => {
                let rt = ret_ty.clone();
                linker
                    .func_new("rayzor", name, func_ty.clone(), move |mut caller, params, results| {
                        let h = val_i32(&params[0]);
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
                        let s = caller.data_mut();
                        let id = s.next_bytes_id;
                        s.next_bytes_id += 1;
                        s.bytes_handles.insert(id, sub);
                        results[0] = ret_int(id, &rt);
                        Ok(())
                    })
                    .map_err(|e| format!("Failed to register {}: {}", name, e))?;
            }

            _ => continue,
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
