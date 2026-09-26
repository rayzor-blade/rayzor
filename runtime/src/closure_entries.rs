//! Alternate entry points for closures, keyed by the code pointer a closure
//! record carries.
//!
//! A closure record is `{code, env}` and its code speaks the declared types:
//! `(env, i32) -> f64` for an `Int->Float` lambda. Two callers cannot know
//! those types: a call through a `Dynamic` value, and the runtime helpers
//! that loop over an array with a callback. The compiler emits a thunk of
//! each shape it needs and registers it here at module init; the callers
//! look the thunk up by the record's code pointer and call that instead.
//!
//! - the *slot* entry is `(env, i64..) -> i64`: every argument and the
//!   result travel as 64-bit slots (a float as its bits);
//! - the *dynamic* entry is `(env, box..) -> box`: every argument and the
//!   result are `DynamicValue` boxes.

use crate::type_system::{
    TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_NULL, dynamic_value_if_boxed, haxe_unbox_bool,
    haxe_unbox_float, haxe_unbox_int,
};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Clone, Copy, Default)]
struct Entries {
    slot: usize,
    dynamic: usize,
}

static ENTRIES: RwLock<Option<HashMap<usize, Entries>>> = RwLock::new(None);

/// The code pointer at offset 0 of a closure record, or 0 for a null record.
unsafe fn record_code(record: *const u8) -> usize {
    if record.is_null() {
        0
    } else {
        unsafe { *(record as *const usize) }
    }
}

/// Register the entries for the closure whose record is `target`. A null
/// `slot` or `dynamic` record leaves that entry unset.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_closure_register_entries(
    target: *const u8,
    slot: *const u8,
    dynamic: *const u8,
) {
    let (code, slot, dynamic) =
        unsafe { (record_code(target), record_code(slot), record_code(dynamic)) };
    if code == 0 {
        return;
    }
    let mut guard = ENTRIES.write().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let entry = map.entry(code).or_default();
    if slot != 0 {
        entry.slot = slot;
    }
    if dynamic != 0 {
        entry.dynamic = dynamic;
    }
}

/// Instance methods reachable by name, `(class type id, name) -> code` of
/// the method's bound thunk: `(env, args..)` with the receiver in `env[0]`.
static METHODS: RwLock<Option<HashMap<(u32, String), usize>>> = RwLock::new(None);

/// Register `record`'s code as the bound thunk of method `name` on class
/// `type_id`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_register_method(type_id: i64, name: *const u8, record: *const u8) {
    let code = unsafe { record_code(record) };
    if code == 0 || name.is_null() {
        return;
    }
    let name = unsafe {
        let hs = &*(name as *const crate::haxe_string::HaxeString);
        if hs.ptr.is_null() {
            return;
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(hs.ptr, hs.len)).into_owned()
    };
    let mut guard = METHODS.write().unwrap();
    guard
        .get_or_insert_with(HashMap::new)
        .insert((type_id as u32, name), code);
}

/// The bound thunk of method `name` on class `type_id`, if registered.
pub(crate) fn method_code(type_id: u32, name: &str) -> Option<usize> {
    let guard = METHODS.read().unwrap();
    guard.as_ref()?.get(&(type_id, name.to_string())).copied()
}

/// A closure record `{code, env}` whose one-slot env holds `receiver`, the
/// shape a compiled bound-method value has.
pub(crate) fn bound_method_record(code: usize, receiver: *mut u8) -> *mut u8 {
    unsafe {
        let env = libc::malloc(8) as *mut usize;
        let record = libc::malloc(16) as *mut usize;
        if env.is_null() || record.is_null() {
            return std::ptr::null_mut();
        }
        *env = receiver as usize;
        *record = code;
        *record.add(1) = env as usize;
        record as *mut u8
    }
}

fn lookup(code: usize) -> Entries {
    let guard = ENTRIES.read().unwrap();
    guard
        .as_ref()
        .and_then(|m| m.get(&code).copied())
        .unwrap_or_default()
}

/// The slot-shaped entry for `code`, or `code` itself when none is
/// registered (a closure whose declared types already are 64-bit slots).
pub(crate) fn closure_slot_code(code: usize) -> usize {
    match lookup(code).slot {
        0 => code,
        slot => slot,
    }
}

/// A view of `closure` whose code is its dynamic entry, for a call whose
/// argument and result types are all `Dynamic`. The closure itself when no
/// entry is registered.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_closure_dynamic_view(closure: *mut u8) -> *mut u8 {
    if closure.is_null() {
        return closure;
    }
    let (code, env) = unsafe {
        (
            *(closure as *const usize),
            *(closure as *const usize).add(1),
        )
    };
    match lookup(code).dynamic {
        0 => closure,
        entry => Box::into_raw(Box::new([entry, env])) as *mut u8,
    }
}

/// A box's payload as a 64-bit slot: an Int or Bool as its value, a Float as
/// its bits, a String or reference as its pointer. A non-box passes through.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_dynamic_to_slot(ptr: *mut u8) -> i64 {
    if ptr.is_null() {
        return 0;
    }
    match dynamic_value_if_boxed(ptr) {
        Some(d) if d.type_id == TYPE_NULL => 0,
        Some(d) if d.type_id == TYPE_INT => haxe_unbox_int(d),
        Some(d) if d.type_id == TYPE_BOOL => haxe_unbox_bool(d) as i64,
        Some(d) if d.type_id == TYPE_FLOAT => haxe_unbox_float(d).to_bits() as i64,
        Some(d) => d.value_ptr as usize as i64,
        None => ptr as usize as i64,
    }
}

/// A box's payload as a float: a Float's value, an Int's conversion.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_dynamic_to_f64(ptr: *mut u8) -> f64 {
    if ptr.is_null() {
        return 0.0;
    }
    match dynamic_value_if_boxed(ptr) {
        Some(d) if d.type_id == TYPE_FLOAT => haxe_unbox_float(d),
        Some(d) if d.type_id == TYPE_INT || d.type_id == TYPE_BOOL => haxe_unbox_int(d) as f64,
        _ => 0.0,
    }
}
