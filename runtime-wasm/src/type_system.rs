//! Class + interface vtable registries for the WASM runtime.
//!
//! Mirrors the native `runtime/src/type_system.rs` surface that
//! `__vtable_init__` (emitted per module by hir_to_mir) and the
//! interface-cast lowering call into. Without these, every
//! registration was a linker return-zero stub, `haxe_iface_fat_ptr_build`
//! returned null at `cast(loaded.model, CausalLanguageModel)`-style
//! sites, and the first interface method dispatch read a garbage slot
//! from a null fat pointer — trapping with "indirect call type
//! mismatch" (or silently mis-dispatching when types happened to
//! coincide).
//!
//! ABI notes (wasm32):
//! - "Closure pointers" are `{ table_slot: i32, env: i32 }` objects in
//!   linear memory. They arrive as i32 (the wasm32 pointer width the
//!   user module's generated `__vtable_init__` actually passes — the
//!   linker requires EXACT type equality with the user import, else it
//!   stubs the function) and are stored zero-extended as i64, the lane
//!   width fat-pointer loads use.
//! - Object headers carry the deterministic class type id as an i64 at
//!   offset 0 (read unaligned — malloc alignment is 8 but views may
//!   not be).
//! - Fat pointers are `[obj: i64][slot_1: i64]…[slot_n: i64]` — the
//!   dispatch site loads `fat + 8*(method_index+1)` as i64.

use std::alloc::{alloc, Layout};
use std::collections::HashMap;
use std::sync::Mutex;

static VTABLES: Mutex<Option<HashMap<u32, Vec<i64>>>> = Mutex::new(None);
static IFACE_VTABLES: Mutex<Option<HashMap<(u32, u32), Vec<i64>>>> = Mutex::new(None);
static IFACE_IMPLS: Mutex<Option<HashMap<i64, Vec<i64>>>> = Mutex::new(None);
static CONSTRUCTORS: Mutex<Option<HashMap<i64, i64>>> = Mutex::new(None);

#[no_mangle]
pub extern "C" fn haxe_vtable_init(type_id: i32, slot_count: i32) {
    let mut g = VTABLES.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    map.insert(type_id as u32, vec![0i64; slot_count.max(0) as usize]);
}

#[no_mangle]
pub extern "C" fn haxe_vtable_set_slot(type_id: i32, slot_index: i32, closure_ptr: i32) {
    let mut g = VTABLES.lock().unwrap();
    if let Some(map) = g.as_mut() {
        if let Some(vt) = map.get_mut(&(type_id as u32)) {
            let idx = slot_index as usize;
            if idx < vt.len() {
                vt[idx] = closure_ptr as u32 as i64;
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn haxe_vtable_lookup(obj_ptr: i32, slot_index: i32) -> i64 {
    if obj_ptr == 0 {
        return 0;
    }
    let type_id = core::ptr::read_unaligned(obj_ptr as *const i64) as u32;
    let g = VTABLES.lock().unwrap();
    if let Some(map) = g.as_ref() {
        if let Some(vt) = map.get(&type_id) {
            let idx = slot_index as usize;
            if idx < vt.len() {
                return vt[idx];
            }
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn haxe_type_register_constructor(type_id: i32, ctor_closure_ptr: i32) {
    let mut g = CONSTRUCTORS.lock().unwrap();
    g.get_or_insert_with(HashMap::new)
        .insert(type_id as i64, ctor_closure_ptr as u32 as i64);
}

#[no_mangle]
pub extern "C" fn haxe_register_interface_impl(class_type_id: i32, iface_type_id: i32) {
    let mut g = IFACE_IMPLS.lock().unwrap();
    g.get_or_insert_with(HashMap::new)
        .entry(class_type_id as i64)
        .or_default()
        .push(iface_type_id as i64);
}

#[no_mangle]
pub extern "C" fn haxe_class_implements(class_type_id: i64, iface_type_id: i64) -> i32 {
    let g = IFACE_IMPLS.lock().unwrap();
    if let Some(map) = g.as_ref() {
        if let Some(list) = map.get(&class_type_id) {
            return list.contains(&iface_type_id) as i32;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn haxe_iface_vtable_set_slot(
    class_type_id: i32,
    iface_type_id: i32,
    slot_index: i32,
    closure_ptr: i32,
) {
    let mut g = IFACE_VTABLES.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    let slots = map
        .entry((class_type_id as u32, iface_type_id as u32))
        .or_default();
    let idx = slot_index as usize;
    if slots.len() <= idx {
        slots.resize(idx + 1, 0);
    }
    slots[idx] = closure_ptr as u32 as i64;
}

/// Build a fresh interface fat pointer `[obj][slot…]` for `obj`'s class
/// and the target interface. Returns 0 (null) when the pair was never
/// registered — callers treat that as a failed cast.
#[no_mangle]
pub unsafe extern "C" fn haxe_iface_fat_ptr_build(obj_ptr: i32, iface_type_id: i32) -> i32 {
    if obj_ptr == 0 {
        return 0;
    }
    let class_type_id = core::ptr::read_unaligned(obj_ptr as *const i64) as u32;
    let g = IFACE_VTABLES.lock().unwrap();
    let Some(map) = g.as_ref() else {
        return 0;
    };
    let Some(vtable) = map.get(&(class_type_id, iface_type_id as u32)) else {
        return 0;
    };
    let total = 8 + 8 * vtable.len();
    let layout = Layout::from_size_align(total, 8).unwrap();
    let fat = alloc(layout);
    if fat.is_null() {
        return 0;
    }
    core::ptr::write_unaligned(fat as *mut i64, obj_ptr as i64);
    for (i, &slot) in vtable.iter().enumerate() {
        core::ptr::write_unaligned(fat.add(8 + 8 * i) as *mut i64, slot);
    }
    fat as i32
}
