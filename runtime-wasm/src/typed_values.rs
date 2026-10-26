//! Typed array access, generic boxing and closure/reflection registries.
//!
//! Ports of the native `haxe_array_*_typed`, `haxe_box_generic_ptr` family
//! and the `closure_entries` / static-field registries onto the wasm32
//! layouts in `lib.rs`:
//! - arrays: the 32-byte header read by `read_array`; slots are
//!   `elem_size` wide (0 means 8). The header has no element-kind flags, so
//!   slots always hold the caller's own representation and the typed reads
//!   and writes move raw bits.
//! - boxes: `{type_id: u32, value_ptr: u32}`; scalars point at a cell.
//!   References box under tag 6, arrays under 7, closures under
//!   `TYPE_FUNCTION`: the tags the compiler hands `haxe_unbox_if_tag`.
//! - values the compiler types I64 arrive as i32 (low word), so Float bits
//!   and the `Null<Int>` null do not survive these signatures.
//! - closure records are `{table_slot: i32 @0, env: i32 @8}`; registries key
//!   and store the table slot.

use crate::{
    NULL_INT, TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_NULL, alloc_dynamic, float_to_buf,
    haxe_box_bool_ptr, haxe_box_float_ptr, haxe_box_int_ptr, haxe_null_int_to_dynamic,
    haxe_unbox_bool, haxe_unbox_float, haxe_unbox_int, haxe_unbox_reference_ptr, int_to_buf,
    read_array, read_dynamic, read_haxe_string, wasm_string_of,
};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

// `ValueTag` values (runtime/src/type_system.rs).
const TAG_UNRESOLVED: i32 = 0;
const TAG_INT: i32 = 1;
const TAG_BOOL: i32 = 2;
const TAG_FLOAT: i32 = 4;
const TAG_STRING: i32 = 5;
const TAG_REFERENCE: i32 = 6;
const TAG_INTERFACE: i32 = 8;
const TAG_FUNCTION: i32 = 9;
const TAG_NULL_INT: i32 = 10;
const TAG_ENUM: i32 = 11;
const TAG_DYNAMIC: i32 = 12;
const TAG_ARRAY: i32 = 13;
const TAG_INT64: i32 = 14;

// Box type ids.
const TYPE_STRING: u32 = 5;
const TYPE_OBJECT: u32 = 6;
const TYPE_ARRAY: u32 = 7;
const TYPE_FUNCTION: u32 = u32::MAX - 1;

/// Slot layout of each array box, keyed by box address, for erased reads.
static ARRAY_BOXES: Mutex<Option<HashMap<u32, u32>>> = Mutex::new(None);

// ---------------------------------------------------------------------------
// Slots
// ---------------------------------------------------------------------------

/// `(data, len, slot width)` of an array header.
unsafe fn array_parts(arr: i32) -> (u32, u32, u32) {
    let (data, len, _, es) = unsafe { read_array(arr) };
    (data, len, if es == 0 { 8 } else { es })
}

/// Bits of slot `idx`: the whole slot when 8 wide, else the sign-extended word.
unsafe fn slot_bits(arr: i32, idx: u32) -> Option<i64> {
    if arr == 0 {
        return None;
    }
    let (data, len, es) = unsafe { array_parts(arr) };
    if idx >= len || data == 0 {
        return None;
    }
    let addr = (data + idx * es) as usize;
    Some(unsafe {
        if es >= 8 {
            core::ptr::read_unaligned(addr as *const i64)
        } else {
            i64::from(core::ptr::read_unaligned(addr as *const i32))
        }
    })
}

/// Drop the last slot, or the first when `!from_end`, keeping slot width.
unsafe fn remove_end(arr: i32, from_end: bool) {
    let (data, len, es) = unsafe { array_parts(arr) };
    if len == 0 {
        return;
    }
    if !from_end && len > 1 {
        unsafe {
            core::ptr::copy(
                (data + es) as *const u8,
                data as *mut u8,
                ((len - 1) * es) as usize,
            )
        };
    }
    unsafe { *(arr as *mut u32).add(2) = len - 1 };
}

// ---------------------------------------------------------------------------
// Boxes
// ---------------------------------------------------------------------------

fn linear_memory_bytes() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        core::arch::wasm32::memory_size(0) * 65536
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        usize::MAX
    }
}

/// `(type_id, value_ptr)` when `p` could address a box.
fn dynamic_box_at(p: i32) -> Option<(u32, u32)> {
    let addr = p as u32 as usize;
    if addr < 0x1000 || addr & 3 != 0 || addr + 8 > linear_memory_bytes() {
        return None;
    }
    Some(unsafe { read_dynamic(p) })
}

/// A tag a box can carry: a builtin or a user type.
fn tag_is_known(type_id: u32) -> bool {
    (TYPE_NULL..=TYPE_ARRAY).contains(&type_id) || type_id > 100
}

fn box_null() -> i32 {
    unsafe { alloc_dynamic(TYPE_NULL, 0) }
}

/// Box `ptr` under `type_id`; null boxes as the null box.
fn box_pointer(type_id: u32, ptr: i32) -> i32 {
    if ptr == 0 {
        box_null()
    } else {
        unsafe { alloc_dynamic(type_id, ptr as u32) }
    }
}

fn box_array(arr: i32, layout: u32) -> i32 {
    let boxed = unsafe { alloc_dynamic(TYPE_ARRAY, arr as u32) };
    if boxed != 0 {
        ARRAY_BOXES
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .insert(boxed as u32, layout);
    }
    boxed
}

fn array_box_layout(boxed: i32) -> Option<u32> {
    ARRAY_BOXES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|boxes| boxes.get(&(boxed as u32)).copied())
}

/// Box `value` by its value tag, as native `haxe_box_typed_ptr`.
fn box_typed(value: i32, tag: i32) -> i32 {
    match tag {
        TAG_DYNAMIC => value,
        TAG_ARRAY if value == 0 => 0,
        TAG_ARRAY => box_array(value, 0),
        TAG_BOOL => haxe_box_bool_ptr((value != 0) as i32),
        TAG_FLOAT => haxe_box_float_ptr(f64::from_bits(value as u32 as u64)),
        TAG_STRING => box_pointer(TYPE_STRING, value),
        TAG_FUNCTION => box_pointer(TYPE_FUNCTION, value),
        TAG_REFERENCE | TAG_ENUM | TAG_INTERFACE => {
            if value == 0 {
                return 0;
            }
            // A `T` bound to Dynamic already holds a box.
            match dynamic_box_at(value) {
                Some((t, _)) if (TYPE_NULL..=TYPE_ARRAY).contains(&t) || t == TYPE_FUNCTION => {
                    value
                }
                _ => unsafe { alloc_dynamic(TYPE_OBJECT, value as u32) },
            }
        }
        // Int, Int64, `Null<Int>` (its null does not fit an i32) and
        // unresolved tags box as Int.
        _ => haxe_box_int_ptr(value),
    }
}

/// One raw slot as the Dynamic its array box's layout describes.
fn box_erased_array_slot(value: i64, layout: u32) -> i32 {
    let ptr = value as i32;
    match layout & 0xff {
        2 => haxe_box_bool_ptr((value != 0) as i32),
        3 if value == NULL_INT => 0,
        3 | 9 => haxe_box_int_ptr(ptr),
        8 => haxe_null_int_to_dynamic(value),
        4 => haxe_box_float_ptr(f64::from_bits(value as u64)),
        5 if ptr != 0 => unsafe { alloc_dynamic(TYPE_STRING, ptr as u32) },
        6 if ptr != 0 => unsafe { alloc_dynamic(TYPE_OBJECT, ptr as u32) },
        7 if ptr != 0 => box_array(ptr, layout >> 8),
        _ => ptr,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_box_generic_ptr(value: i32, type_tag: i32, metadata: i32) -> i32 {
    let metadata = metadata as u32;
    match metadata & 0xff {
        7 if value != 0 => box_array(value, metadata >> 8),
        6 if value != 0 => unsafe { alloc_dynamic(TYPE_OBJECT, value as u32) },
        _ => box_typed(value, type_tag),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_box_haxestring_ptr(hs_ptr: i32) -> i32 {
    box_pointer(TYPE_STRING, hs_ptr)
}

/// Unwrap a box carrying `tag`, or pass anything else through unchanged.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_unbox_if_tag(ptr: i32, tag: i32) -> i32 {
    let tag = tag as u32;
    match dynamic_box_at(ptr) {
        Some((t, v)) if t == tag => v as i32,
        Some((TYPE_NULL, _)) => 0,
        // A scalar read as a String is its string, as `Std.string` spells it.
        Some((t, _)) if tag == TYPE_STRING && matches!(t, TYPE_INT | TYPE_FLOAT | TYPE_BOOL) => {
            wasm_string_of(&dynamic_text(ptr, 0))
        }
        _ => ptr,
    }
}

/// An erased generic return as the bits the caller's `T` expects.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_unbox_generic_return(ptr: i32, type_tag: i32) -> i32 {
    if type_tag == TAG_DYNAMIC || ptr == 0 {
        return ptr;
    }
    let Some((t, v)) = dynamic_box_at(ptr).filter(|&(t, _)| tag_is_known(t)) else {
        return ptr;
    };
    let reference_box = matches!(t, TYPE_STRING | TYPE_OBJECT | TYPE_ARRAY | TYPE_FUNCTION);
    // A reference reader's raw enum cell can begin with a scalar box's tag.
    let reference_reader = matches!(
        type_tag,
        TAG_STRING | TAG_REFERENCE | TAG_INTERFACE | TAG_FUNCTION | TAG_ENUM | TAG_ARRAY
    );
    match t {
        TYPE_NULL => 0,
        _ if reference_box => v as i32,
        _ if reference_reader => ptr,
        TYPE_INT | TYPE_BOOL => haxe_unbox_int(ptr),
        TYPE_FLOAT => haxe_unbox_float(ptr).to_bits() as i32,
        _ => ptr,
    }
}

// ---------------------------------------------------------------------------
// Typed array access
// ---------------------------------------------------------------------------

/// Only a Dynamic element tag changes native storage (boxed slots); wasm
/// headers have no flag word, so slots stay in the caller's representation.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_init_element_tag(_arr: i32, _tag: i32) {}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_get_typed(arr: i32, index: i32, _tag: i32) -> i32 {
    if index < 0 {
        return 0;
    }
    unsafe { slot_bits(arr, index as u32) }.map_or(0, |bits| bits as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_set_typed(
    arr: i32,
    index: i32,
    bits: i32,
    tag: i32,
    _metadata: i32,
) -> i32 {
    if arr == 0 || index < 0 {
        return 0;
    }
    if tag == TAG_NULL_INT {
        crate::haxe_array_set_null_int(arr, index, i64::from(bits));
    } else {
        crate::haxe_array_set_i64(arr, index, bits);
    }
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_push_typed(arr: i32, bits: i32, _tag: i32, _metadata: i32) -> i32 {
    if arr == 0 {
        return 0;
    }
    crate::haxe_array_push_i64(arr, bits);
    unsafe { array_parts(arr).1 as i32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_push_typed_f64(arr: i32, value: f64) -> i32 {
    if arr == 0 {
        return 0;
    }
    crate::haxe_array_push_any_f64(arr, value);
    unsafe { array_parts(arr).1 as i32 }
}

/// `pop` (`from_end`) or `shift`: the removed slot's bits, 0 when empty.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_take_typed(arr: i32, from_end: i32, tag: i32) -> i32 {
    if arr == 0 {
        return 0;
    }
    let len = unsafe { array_parts(arr).1 };
    if len == 0 {
        return 0;
    }
    let from_end = from_end != 0;
    let index = if from_end { len - 1 } else { 0 };
    let value = haxe_array_get_typed(arr, index as i32, tag);
    unsafe { remove_end(arr, from_end) };
    value
}

/// Index an erased array; a boxed one boxes raw slots by its retained layout.
/// `target`: 0 Dynamic, 1 Int, 2 Float, 3 Bool, 5 reference.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_get_erased(array: i32, index: i32, target: i32) -> i32 {
    if array == 0 || index < 0 {
        return 0;
    }
    let (arr, layout) = match dynamic_box_at(array) {
        Some((TYPE_ARRAY, inner)) => (inner as i32, array_box_layout(array)),
        _ => (array, None),
    };
    let Some(value) = (unsafe { slot_bits(arr, index as u32) }) else {
        return 0;
    };
    let Some(layout) = layout.filter(|&l| l != 0) else {
        return value as i32;
    };
    let boxed = box_erased_array_slot(value, layout);
    if boxed == 0 {
        return 0;
    }
    match target {
        1 => haxe_unbox_int(boxed),
        2 => haxe_unbox_float(boxed).to_bits() as i32,
        3 => haxe_unbox_bool(boxed),
        5 => haxe_unbox_reference_ptr(boxed),
        _ => boxed,
    }
}

// ---------------------------------------------------------------------------
// Stringification
// ---------------------------------------------------------------------------

const MAX_DEPTH: u32 = 8;

fn push_int(out: &mut Vec<u8>, value: i32) {
    let mut buf = [0u8; 12];
    out.extend_from_slice(int_to_buf(value, &mut buf));
}

fn push_float(out: &mut Vec<u8>, value: f64) {
    let mut buf = [0u8; 32];
    let len = float_to_buf(value, &mut buf);
    out.extend_from_slice(&buf[..len]);
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.extend_from_slice(if value { b"true" } else { b"false" });
}

/// The bytes of HaxeString `s`; nothing for a null or empty string.
fn push_haxe_string(out: &mut Vec<u8>, s: i32) {
    if s == 0 {
        return;
    }
    let (ptr, len, _) = unsafe { read_haxe_string(s) };
    if ptr != 0 && len > 0 {
        out.extend_from_slice(unsafe {
            core::slice::from_raw_parts(ptr as *const u8, len as usize)
        });
    }
}

/// `Std.string` of a Dynamic. Without class RTTI an unboxed value prints as
/// its Int, the native fallback for a tag outside the space.
fn dynamic_text(p: i32, depth: u32) -> Vec<u8> {
    let mut out = Vec::new();
    if p == 0 {
        out.extend_from_slice(b"null");
        return out;
    }
    match dynamic_box_at(p).filter(|&(t, _)| tag_is_known(t)) {
        Some((TYPE_NULL, _)) => out.extend_from_slice(b"null"),
        Some((TYPE_BOOL, _)) => push_bool(&mut out, haxe_unbox_bool(p) != 0),
        Some((TYPE_INT, _)) => push_int(&mut out, haxe_unbox_int(p)),
        Some((TYPE_FLOAT, _)) => push_float(&mut out, haxe_unbox_float(p)),
        Some((TYPE_STRING, v)) => push_haxe_string(&mut out, v as i32),
        Some((TYPE_FUNCTION, _)) => out.extend_from_slice(b"<function>"),
        Some((TYPE_ARRAY, v)) => {
            let layout = array_box_layout(p).unwrap_or(0);
            array_text(&mut out, v as i32, layout, depth)
        }
        _ => push_int(&mut out, p),
    }
    out
}

/// `[a,b,c]` of an array whose slots `layout` describes.
fn array_text(out: &mut Vec<u8>, arr: i32, layout: u32, depth: u32) {
    if depth >= MAX_DEPTH {
        out.extend_from_slice(b"[...]");
        return;
    }
    out.push(b'[');
    let len = if arr == 0 {
        0
    } else {
        unsafe { array_parts(arr).1 }
    };
    for i in 0..len {
        if i > 0 {
            out.push(b',');
        }
        let bits = unsafe { slot_bits(arr, i) }.unwrap_or(0);
        match layout & 0xff {
            2 => push_bool(out, bits != 0),
            3 | 8 if bits == NULL_INT => out.extend_from_slice(b"null"),
            3 | 8 | 9 => push_int(out, bits as i32),
            4 => push_float(out, f64::from_bits(bits as u64)),
            5 if bits as i32 == 0 => out.extend_from_slice(b"null"),
            5 => push_haxe_string(out, bits as i32),
            7 if bits as i32 == 0 => out.extend_from_slice(b"null"),
            7 => array_text(out, bits as i32, layout >> 8, depth + 1),
            _ => out.extend_from_slice(&dynamic_text(bits as i32, depth + 1)),
        }
    }
    out.push(b']');
}

/// One slot's text under its element tag, as native `haxe_value_to_string_by_tag`.
fn slot_text(out: &mut Vec<u8>, bits: i64, tag: i32) {
    let value = bits as i32;
    match tag {
        TAG_STRING => {
            // A String inferred through an initially Dynamic array may be boxed.
            let s = match dynamic_box_at(value) {
                Some((TYPE_STRING, v)) => v as i32,
                _ => value,
            };
            push_haxe_string(out, s);
        }
        TAG_INT | TAG_NULL_INT if bits == NULL_INT => out.extend_from_slice(b"null"),
        TAG_INT | TAG_NULL_INT | TAG_INT64 | TAG_UNRESOLVED => push_int(out, value),
        TAG_BOOL => push_bool(out, value != 0),
        TAG_FLOAT => push_float(out, f64::from_bits(bits as u64)),
        TAG_ARRAY if value == 0 => out.extend_from_slice(b"null"),
        TAG_ARRAY => array_text(out, value, 0, 1),
        TAG_REFERENCE | TAG_DYNAMIC | TAG_ENUM | TAG_INTERFACE => {
            out.extend_from_slice(&dynamic_text(value, 1))
        }
        _ => push_int(out, value),
    }
}

/// Type-aware `Array.join`: each slot is stringified by `elem_tag`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_join_typed(arr: i32, sep: i32, elem_tag: i32) -> i32 {
    let mut out = Vec::new();
    let len = if arr == 0 {
        0
    } else {
        unsafe { array_parts(arr).1 }
    };
    let mut separator = Vec::new();
    push_haxe_string(&mut separator, sep);
    for i in 0..len {
        if i > 0 {
            out.extend_from_slice(&separator);
        }
        slot_text(
            &mut out,
            unsafe { slot_bits(arr, i) }.unwrap_or(0),
            elem_tag,
        );
    }
    wasm_string_of(&out)
}

/// `Std.string` of an `Array<E>` of enum `type_id`: heap values when `boxed`,
/// discriminants otherwise. The wasm runtime has no enum RTTI, so values print
/// as native does for an unregistered enum.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_to_string_enum(arr: i32, type_id: i32, boxed: i32) -> i32 {
    let mut out = vec![b'['];
    let len = if arr == 0 {
        0
    } else {
        unsafe { array_parts(arr).1 }
    };
    for i in 0..len {
        if i > 0 {
            out.push(b',');
        }
        let slot = unsafe { slot_bits(arr, i) }.unwrap_or(0);
        if boxed != 0 {
            let cell = slot as i32;
            if cell == 0 {
                out.extend_from_slice(b"null");
            } else {
                let variant = unsafe { core::ptr::read_unaligned(cell as *const i32) };
                out.extend_from_slice(b"<enum ");
                out.extend_from_slice((type_id as u32).to_string().as_bytes());
                out.extend_from_slice(b"::");
                push_int(&mut out, variant);
                out.push(b'>');
            }
        } else if slot == NULL_INT {
            out.extend_from_slice(b"null");
        } else {
            push_int(&mut out, slot as i32);
        }
    }
    out.push(b']');
    wasm_string_of(&out)
}

// ---------------------------------------------------------------------------
// Closure entries and reflection registries
// ---------------------------------------------------------------------------

/// Alternate entries of one closure code, as table slots; 0 is unset.
#[derive(Clone, Copy, Default, Debug)]
pub struct ClosureEntries {
    /// `(env, i64..) -> i64`.
    pub slot: i32,
    /// `(env, box..) -> box`.
    pub dynamic: i32,
    /// `(env, Array<Dynamic>) -> box`.
    pub packed: i32,
}

static ENTRIES: Mutex<Option<HashMap<i32, ClosureEntries>>> = Mutex::new(None);
/// `(class type id, name) -> table slot` of the method's bound thunk.
static METHODS: Mutex<Option<HashMap<(u32, String), i32>>> = Mutex::new(None);
/// Table slots of bound-method thunks, whose env is `[receiver]`.
static BOUND_THUNKS: Mutex<Option<HashSet<i32>>> = Mutex::new(None);
/// `(class type id, name) -> (getter, setter)` table slots; 0 is absent.
static STATIC_FIELDS: Mutex<Option<HashMap<(u32, String), (i32, i32)>>> = Mutex::new(None);

/// The table slot at offset 0 of a closure record, or 0 for a null record.
fn record_code(record: i32) -> i32 {
    if record == 0 {
        0
    } else {
        unsafe { core::ptr::read_unaligned(record as *const i32) }
    }
}

/// A HaxeString's text, lossily decoded; `None` for a null or empty name.
fn name_text(s: i32) -> Option<String> {
    let mut bytes = Vec::new();
    push_haxe_string(&mut bytes, s);
    (!bytes.is_empty()).then(|| String::from_utf8_lossy(&bytes).into_owned())
}

/// Register the entries for the closure whose record is `target`; a null
/// `slot` or `dynamic` record leaves that entry unset.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_closure_register_entries(target: i32, slot: i32, dynamic: i32) {
    let (code, slot, dynamic) = (record_code(target), record_code(slot), record_code(dynamic));
    if code == 0 {
        return;
    }
    let mut guard = ENTRIES.lock().unwrap();
    let entry = guard
        .get_or_insert_with(HashMap::new)
        .entry(code)
        .or_default();
    if slot != 0 {
        entry.slot = slot;
    }
    if dynamic != 0 {
        entry.dynamic = dynamic;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_closure_register_packed_entry(target: i32, packed: i32) {
    let (code, packed) = (record_code(target), record_code(packed));
    if code != 0 && packed != 0 {
        ENTRIES
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(code)
            .or_default()
            .packed = packed;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_closure_register_bound_thunk(record: i32) {
    let code = record_code(record);
    if code != 0 {
        BOUND_THUNKS
            .lock()
            .unwrap()
            .get_or_insert_with(HashSet::new)
            .insert(code);
    }
}

/// Register `record`'s code as the bound thunk of method `name` on `type_id`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_register_method(type_id: i32, name: i32, record: i32) {
    let code = record_code(record);
    let Some(name) = (code != 0).then(|| name_text(name)).flatten() else {
        return;
    };
    METHODS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert((type_id as u32, name), code);
}

/// Record a class static's accessors: `getter(env) -> box`, `setter(env, box)`;
/// either record may be null.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_register_static_field(type_id: i32, name: i32, getter: i32, setter: i32) {
    // A name typed Dynamic arrives in its String box.
    let name = match dynamic_box_at(name) {
        Some((TYPE_STRING, v)) => v as i32,
        _ => name,
    };
    let Some(name) = name_text(name) else {
        return;
    };
    STATIC_FIELDS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(
            (type_id as u32, name),
            (record_code(getter), record_code(setter)),
        );
}

/// The registered entries for closure code `code` (a table slot).
pub fn closure_entries(code: i32) -> ClosureEntries {
    let guard = ENTRIES.lock().unwrap();
    guard
        .as_ref()
        .and_then(|m| {
            m.get(&code).copied().or_else(|| {
                m.values()
                    .find(|e| e.slot == code || e.dynamic == code || e.packed == code)
                    .copied()
            })
        })
        .unwrap_or_default()
}

fn closure_view(record: i32, entry: impl FnOnce(ClosureEntries) -> i32) -> i32 {
    let code = record_code(record);
    if code == 0 {
        return record;
    }
    let target = entry(closure_entries(code));
    if target == 0 || target == code {
        return record;
    }
    let view = crate::rt_alloc(16);
    if view != 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(record as *const u8, view as *mut u8, 16);
            core::ptr::write_unaligned(view as *mut i64, target as u32 as i64);
        }
    }
    view
}

/// A closure record using its slot-shaped entry, retaining its environment.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_closure_slot_view(record: i32) -> i32 {
    closure_view(record, |entry| entry.slot)
}

/// A closure record using its Dynamic-shaped entry, retaining its environment.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_closure_dynamic_view(record: i32) -> i32 {
    closure_view(record, |entry| entry.dynamic)
}

/// The bound thunk of method `name` on class `type_id`, if registered.
pub fn method_code(type_id: u32, name: &str) -> Option<i32> {
    let guard = METHODS.lock().unwrap();
    guard.as_ref()?.get(&(type_id, name.to_string())).copied()
}

pub fn is_bound_thunk(code: i32) -> bool {
    BOUND_THUNKS
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|set| set.contains(&code))
}

/// `(getter, setter)` table slots of static `name` on class `type_id`.
pub fn static_accessors(type_id: u32, name: &str) -> Option<(i32, i32)> {
    let guard = STATIC_FIELDS.lock().unwrap();
    guard.as_ref()?.get(&(type_id, name.to_string())).copied()
}

/// The `(env, slot..) -> slot` entry of closure code `code`, as a table index.
fn slot_code(code: i32) -> i32 {
    match closure_entries(code).slot {
        0 => code,
        slot => slot,
    }
}

/// Slot `i` of `(data, elem_size)`, widened to 64 bits.
unsafe fn read_slot(data: u32, es: u32, i: u32) -> u64 {
    unsafe {
        let p = data + i * es;
        if es >= 8 {
            *(p as *const u64)
        } else {
            *(p as *const u32) as u64
        }
    }
}

unsafe fn write_slot(data: u32, es: u32, i: u32, v: u64) {
    unsafe {
        let p = data + i * es;
        if es >= 8 {
            *(p as *mut u64) = v;
        } else {
            *(p as *mut u32) = v as u32;
        }
    }
}

/// `Array.sort(cmp)`: stable insertion sort, as the native runtime.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_sort(arr: i32, fn_ptr: i32, env: i32) {
    if arr == 0 || fn_ptr == 0 {
        return;
    }
    let cmp: extern "C" fn(i32, i32, i32) -> i32 =
        unsafe { core::mem::transmute(slot_code(fn_ptr) as usize) };
    unsafe {
        let (data, len, _, es) = read_array(arr);
        let es = if es == 0 { 8 } else { es };
        for i in 1..len {
            let key = read_slot(data, es, i);
            let mut j = i;
            while j > 0 {
                let prev = read_slot(data, es, j - 1);
                if cmp(env, prev as u32 as i32, key as u32 as i32) > 0 {
                    write_slot(data, es, j, prev);
                    j -= 1;
                } else {
                    break;
                }
            }
            write_slot(data, es, j, key);
        }
    }
}

/// Copies a fresh array's header into the caller's `out` header.
unsafe fn emit_into(out: i32, result: i32) {
    if out != 0 && result != 0 {
        unsafe {
            let (dp, l, c, es) = read_array(result);
            crate::write_array(out, dp, l, c, es);
        }
    }
}

/// `Array.map(f)` into the header at `out`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_map(out: i32, arr: i32, fn_ptr: i32, env: i32) {
    let result = crate::haxe_array_new();
    if arr != 0 && fn_ptr != 0 {
        let f: extern "C" fn(i32, i32) -> i32 =
            unsafe { core::mem::transmute(slot_code(fn_ptr) as usize) };
        unsafe {
            let (data, len, _, es) = read_array(arr);
            let es = if es == 0 { 8 } else { es };
            for i in 0..len {
                crate::haxe_array_push_i64(result, f(env, read_slot(data, es, i) as u32 as i32));
            }
        }
    }
    unsafe { emit_into(out, result) }
}

/// `Array.filter(f)` into the header at `out`; kept elements keep their slot.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_filter(out: i32, arr: i32, fn_ptr: i32, env: i32) {
    let result = crate::haxe_array_new();
    if arr != 0 && fn_ptr != 0 {
        let f: extern "C" fn(i32, i32) -> i32 =
            unsafe { core::mem::transmute(slot_code(fn_ptr) as usize) };
        unsafe {
            let (data, len, _, es) = read_array(arr);
            let es = if es == 0 { 8 } else { es };
            for i in 0..len {
                let v = read_slot(data, es, i);
                if f(env, v as u32 as i32) != 0 {
                    crate::haxe_array_push_i64(result, v as u32 as i32);
                    let (rd, rl, _, res) = read_array(result);
                    write_slot(rd, if res == 0 { 8 } else { res }, rl - 1, v);
                }
            }
        }
    }
    unsafe { emit_into(out, result) }
}

/// Releases an array's element buffer, which the runtime allocated as
/// `cap * elem_size` bytes; the header is the caller's to free.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_free(arr: i32) {
    if arr == 0 {
        return;
    }
    unsafe {
        let (data, _, cap, es) = read_array(arr);
        let es = if es == 0 { 8 } else { es };
        if data != 0 && cap > 0 {
            let layout = std::alloc::Layout::from_size_align_unchecked((cap * es) as usize, 8);
            std::alloc::dealloc(data as *mut u8, layout);
        }
    }
}

/// `[a,b]` with each slot rendered by `render`.
fn array_to_string_with(arr: i32, render: impl Fn(u32, u32, u32, &mut Vec<u8>)) -> i32 {
    let mut s = vec![b'['];
    if arr != 0 {
        unsafe {
            let (data, len, _, es) = read_array(arr);
            let es = if es == 0 { 8 } else { es };
            for i in 0..len {
                if i > 0 {
                    s.push(b',');
                }
                render(data, es, i, &mut s);
            }
        }
    }
    s.push(b']');
    crate::wasm_string_of(&s)
}

/// `toString` of an `Array<Int>` / `Array<Null<Int>>`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_to_string_i64(arr: i32) -> i32 {
    array_to_string_with(arr, |data, es, i, s| {
        let v = unsafe { read_slot(data, es, i) };
        if es >= 8 && v == crate::NULL_INT as u64 {
            s.extend_from_slice(b"null");
        } else {
            let mut buf = [0u8; 12];
            s.extend_from_slice(crate::int_to_buf(v as u32 as i32, &mut buf));
        }
    })
}

/// `toString` of an `Array<Float>`; 4-byte slots hold f32.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_array_to_string_f64(arr: i32) -> i32 {
    array_to_string_with(arr, |data, es, i, s| {
        let v = unsafe { read_slot(data, es, i) };
        let f = if es >= 8 {
            f64::from_bits(v)
        } else {
            f32::from_bits(v as u32) as f64
        };
        let mut buf = [0u8; 32];
        let n = crate::float_to_buf(f, &mut buf);
        s.extend_from_slice(&buf[..n]);
    })
}
