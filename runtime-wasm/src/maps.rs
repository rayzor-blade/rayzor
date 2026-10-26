//! `haxe.ds.StringMap`, `IntMap` and `ObjectMap` for wasm32.
//!
//! Values are raw slot words, the i32 an array element holds on wasm32. A map
//! remembers the layout its first typed write declared, so typed and Dynamic
//! views share entries. Like native maps and wasm arrays, a map takes no lock:
//! Haxe maps are not thread-safe, and the allocator behind them is.

use crate::{
    NULL_INT, TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_NULL, float_to_buf, haxe_array_new,
    haxe_array_push_i64, haxe_box_bool_ptr, haxe_box_int_ptr, haxe_coerce_dynamic_to_int,
    haxe_null_int_from_dynamic, haxe_unbox_bool, read_dynamic, read_haxe_string, wasm_string_of,
};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const TAG_INT: i32 = 1;
const TAG_BOOL: i32 = 2;
const TAG_STRING: i32 = 5;
const TAG_NULL_INT: i32 = 10;
const TAG_DYNAMIC: i32 = 12;

/// FNV-1a, so iteration order is the same on every run.
struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for Fnv {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type Table<K> = HashMap<K, i32, BuildHasherDefault<Fnv>>;

#[derive(Clone, Copy, Default)]
struct ValueLayout {
    tag: i32,
}

impl ValueLayout {
    /// `bits` stored under `self`, re-encoded for `target`.
    fn transcode(self, bits: i32, target: ValueLayout) -> i32 {
        // Int and Null<Int> share the slot encoding on wasm32.
        let int_like = |t: i32| t == TAG_INT || t == TAG_NULL_INT;
        if self.tag == 0 || self.tag == target.tag || (int_like(self.tag) && int_like(target.tag)) {
            return bits;
        }
        match (self.tag, target.tag) {
            (s, TAG_DYNAMIC) if int_like(s) => haxe_box_int_ptr(bits),
            (TAG_BOOL, TAG_DYNAMIC) => haxe_box_bool_ptr(bits),
            (TAG_DYNAMIC, t) if int_like(t) => haxe_coerce_dynamic_to_int(bits),
            (TAG_DYNAMIC, TAG_BOOL) => haxe_unbox_bool(bits),
            _ => bits,
        }
    }

    /// Adopt `source` if no layout is set yet, then encode `bits` for it.
    fn write(&mut self, bits: i32, source: ValueLayout) -> i32 {
        if self.tag == 0 {
            *self = source;
        }
        source.transcode(bits, *self)
    }

    fn render(self, bits: i32) -> String {
        match self.tag {
            TAG_BOOL => (if bits != 0 { "true" } else { "false" }).to_string(),
            TAG_STRING => unsafe { string_text(bits) },
            TAG_DYNAMIC => unsafe { dynamic_text(bits) },
            _ => bits.to_string(),
        }
    }

    /// `bits` read through a static value type `tag`, rendered as that type.
    fn render_as(self, bits: i32, tag: i32) -> String {
        if tag == 0 {
            return bits.to_string();
        }
        let target = ValueLayout { tag };
        target.render(self.transcode(bits, target))
    }
}

/// A Dynamic box rendered as its value; other references print as their address.
unsafe fn dynamic_text(ptr: i32) -> String {
    if ptr == 0 {
        return "null".to_string();
    }
    let (type_id, value_ptr) = unsafe { read_dynamic(ptr) };
    match type_id {
        TYPE_NULL => "null".to_string(),
        TYPE_BOOL => (if unsafe { *(value_ptr as *const i32) } != 0 {
            "true"
        } else {
            "false"
        })
        .to_string(),
        TYPE_INT => unsafe { *(value_ptr as *const i32) }.to_string(),
        TYPE_FLOAT => {
            let mut buf = [0u8; 32];
            let n = float_to_buf(unsafe { *(value_ptr as *const f64) }, &mut buf);
            String::from_utf8_lossy(&buf[..n]).into_owned()
        }
        _ => ptr.to_string(),
    }
}

struct Map<K> {
    table: Table<K>,
    layout: ValueLayout,
}

impl<K: Clone> Map<K> {
    fn alloc() -> i32 {
        Box::into_raw(Box::new(Map::<K> {
            table: Table::default(),
            layout: ValueLayout::default(),
        })) as i32
    }

    unsafe fn at<'a>(ptr: i32) -> Option<&'a mut Map<K>> {
        if ptr == 0 {
            None
        } else {
            Some(unsafe { &mut *(ptr as *mut Map<K>) })
        }
    }

    unsafe fn copy(ptr: i32) -> i32 {
        let Some(map) = (unsafe { Self::at(ptr) }) else {
            return Self::alloc();
        };
        Box::into_raw(Box::new(Map::<K> {
            table: map.table.clone(),
            layout: map.layout,
        })) as i32
    }

    unsafe fn values_to_array(ptr: i32) -> i32 {
        let arr = haxe_array_new();
        if let Some(map) = unsafe { Self::at(ptr) } {
            for &v in map.table.values() {
                haxe_array_push_i64(arr, v);
            }
        }
        arr
    }

    /// The value for `bits`, re-encoded for the static type `tag`; 0 if absent.
    fn read_typed(&self, bits: Option<&i32>, tag: i32) -> i32 {
        bits.map_or(0, |&b| self.layout.transcode(b, ValueLayout { tag }))
    }

    unsafe fn count(ptr: i32) -> i32 {
        unsafe { Self::at(ptr) }.map_or(0, |m| m.table.len() as i32)
    }

    /// `val_tag` is the value's static type for the typed renderers; `None`
    /// renders by the stored layout.
    unsafe fn to_string(
        ptr: i32,
        empty: &str,
        key: impl Fn(&K) -> String,
        val_tag: Option<i32>,
    ) -> i32 {
        let Some(map) = (unsafe { Self::at(ptr) }) else {
            return wasm_string_of(empty.as_bytes());
        };
        let value = |v: i32| match val_tag {
            Some(tag) => map.layout.render_as(v, tag),
            None => map.layout.render(v),
        };
        let entries: Vec<String> = map
            .table
            .iter()
            .map(|(k, v)| format!("{} => {}", key(k), value(*v)))
            .collect();
        wasm_string_of(format!("[{}]", entries.join(", ")).as_bytes())
    }
}

/// The bytes of HaxeString `s`; `None` for a null string.
unsafe fn key_bytes<'a>(s: i32) -> Option<&'a [u8]> {
    if s == 0 {
        return None;
    }
    let (ptr, len, _) = unsafe { read_haxe_string(s) };
    if ptr == 0 || len == 0 {
        return Some(&[]);
    }
    Some(unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) })
}

unsafe fn string_text(s: i32) -> String {
    match unsafe { key_bytes(s) } {
        Some(b) => String::from_utf8_lossy(b).into_owned(),
        None => "null".to_string(),
    }
}

type StrMap = Map<Box<[u8]>>;
type IntMap = Map<i32>;
type ObjMap = Map<i32>;

fn str_key_text(k: &[u8]) -> String {
    String::from_utf8_lossy(k).into_owned()
}

/// Leak `words` as a buffer of 8-byte slots, the stride wasm code reads
/// `Ptr<I64>` elements at, storing its length in the i64 at `out_len`.
unsafe fn leak_words(words: Vec<i64>, out_len: i32) -> i32 {
    if out_len != 0 {
        unsafe { core::ptr::write_unaligned(out_len as *mut i64, words.len() as i64) };
    }
    if words.is_empty() {
        return 0;
    }
    Box::into_raw(words.into_boxed_slice()) as *mut i64 as i32
}

fn str_insert(map: &mut StrMap, key: &[u8], value: i32) {
    match map.table.get_mut(key) {
        Some(slot) => *slot = value,
        None => {
            map.table.insert(key.into(), value);
        }
    }
}

// ============================================================================
// StringMap
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_new() -> i32 {
    StrMap::alloc()
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_set(map: i32, key: i32, value: i32) {
    unsafe {
        if let (Some(m), Some(k)) = (StrMap::at(map), key_bytes(key)) {
            str_insert(m, k, value);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_set_typed(
    map: i32,
    key: i32,
    value: i32,
    tag: i32,
    _metadata: i32,
) {
    unsafe {
        if let (Some(m), Some(k)) = (StrMap::at(map), key_bytes(key)) {
            let value = m.layout.write(value, ValueLayout { tag });
            str_insert(m, k, value);
        }
    }
}

/// The value, or 0 for an absent key.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_get(map: i32, key: i32) -> i32 {
    unsafe {
        match (StrMap::at(map), key_bytes(key)) {
            (Some(m), Some(k)) => m.table.get(k).copied().unwrap_or(0),
            _ => 0,
        }
    }
}

/// The value read as `Null<Int>`: the Int, or `NULL_INT` for an absent key.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_get_null_int(map: i32, key: i32, tag: i32, _metadata: i32) -> i64 {
    unsafe {
        let (Some(m), Some(k)) = (StrMap::at(map), key_bytes(key)) else {
            return NULL_INT;
        };
        m.table
            .get(k)
            .map_or(NULL_INT, |&bits| null_int_of(m.layout, bits, tag))
    }
}

fn null_int_of(layout: ValueLayout, bits: i32, tag: i32) -> i64 {
    if layout.tag == TAG_DYNAMIC {
        return haxe_null_int_from_dynamic(bits);
    }
    i64::from(layout.transcode(bits, ValueLayout { tag }))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_exists(map: i32, key: i32) -> i32 {
    unsafe {
        match (StrMap::at(map), key_bytes(key)) {
            (Some(m), Some(k)) => m.table.contains_key(k) as i32,
            _ => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_remove(map: i32, key: i32) -> i32 {
    unsafe {
        match (StrMap::at(map), key_bytes(key)) {
            (Some(m), Some(k)) => m.table.remove(k).is_some() as i32,
            _ => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_clear(map: i32) {
    if let Some(m) = unsafe { StrMap::at(map) } {
        m.table.clear();
    }
}

/// Shallow copy: same keys, same value words.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_copy(map: i32) -> i32 {
    unsafe { StrMap::copy(map) }
}

/// Keys as an array of fresh HaxeStrings.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_keys_to_array(map: i32) -> i32 {
    let arr = haxe_array_new();
    if let Some(m) = unsafe { StrMap::at(map) } {
        for k in m.table.keys() {
            haxe_array_push_i64(arr, wasm_string_of(k));
        }
    }
    arr
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_values_to_array(map: i32) -> i32 {
    unsafe { StrMap::values_to_array(map) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_to_string(map: i32) -> i32 {
    unsafe { StrMap::to_string(map, "{}", |k| str_key_text(k), None) }
}

// ============================================================================
// IntMap
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_new() -> i32 {
    IntMap::alloc()
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_set(map: i32, key: i32, value: i32) {
    if let Some(m) = unsafe { IntMap::at(map) } {
        m.table.insert(key, value);
    }
}

/// The value, or 0 for an absent key.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_get(map: i32, key: i32) -> i32 {
    unsafe { IntMap::at(map) }.map_or(0, |m| m.table.get(&key).copied().unwrap_or(0))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_exists(map: i32, key: i32) -> i32 {
    unsafe { IntMap::at(map) }.is_some_and(|m| m.table.contains_key(&key)) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_remove(map: i32, key: i32) -> i32 {
    unsafe { IntMap::at(map) }.is_some_and(|m| m.table.remove(&key).is_some()) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_clear(map: i32) {
    if let Some(m) = unsafe { IntMap::at(map) } {
        m.table.clear();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_copy(map: i32) -> i32 {
    unsafe { IntMap::copy(map) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_keys_to_array(map: i32) -> i32 {
    let arr = haxe_array_new();
    if let Some(m) = unsafe { IntMap::at(map) } {
        for &k in m.table.keys() {
            haxe_array_push_i64(arr, k);
        }
    }
    arr
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_values_to_array(map: i32) -> i32 {
    unsafe { IntMap::values_to_array(map) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_to_string(map: i32) -> i32 {
    unsafe { IntMap::to_string(map, "{}", |k| k.to_string(), None) }
}

// ============================================================================
// ObjectMap: keys compared by pointer identity
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_new() -> i32 {
    ObjMap::alloc()
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_set(map: i32, key: i32, value: i32) {
    if let Some(m) = unsafe { ObjMap::at(map) } {
        m.table.insert(key, value);
    }
}

/// The value, or 0 for an absent key.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_get(map: i32, key: i32) -> i32 {
    unsafe { ObjMap::at(map) }.map_or(0, |m| m.table.get(&key).copied().unwrap_or(0))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_exists(map: i32, key: i32) -> i32 {
    unsafe { ObjMap::at(map) }.is_some_and(|m| m.table.contains_key(&key)) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_remove(map: i32, key: i32) -> i32 {
    unsafe { ObjMap::at(map) }.is_some_and(|m| m.table.remove(&key).is_some()) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_clear(map: i32) {
    if let Some(m) = unsafe { ObjMap::at(map) } {
        m.table.clear();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_copy(map: i32) -> i32 {
    unsafe { ObjMap::copy(map) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_keys_to_array(map: i32) -> i32 {
    let arr = haxe_array_new();
    if let Some(m) = unsafe { ObjMap::at(map) } {
        for &k in m.table.keys() {
            haxe_array_push_i64(arr, k);
        }
    }
    arr
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_values_to_array(map: i32) -> i32 {
    unsafe { ObjMap::values_to_array(map) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_to_string(map: i32) -> i32 {
    unsafe { ObjMap::to_string(map, "null", |k| format!("0x{:x}", *k as u32), None) }
}

// ============================================================================
// Typed views, counts and key buffers
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_get_typed(map: i32, key: i32, tag: i32, _metadata: i32) -> i32 {
    unsafe {
        match (StrMap::at(map), key_bytes(key)) {
            (Some(m), Some(k)) => m.read_typed(m.table.get(k), tag),
            _ => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_set_typed(map: i32, key: i32, value: i32, tag: i32, _metadata: i32) {
    if let Some(m) = unsafe { IntMap::at(map) } {
        let value = m.layout.write(value, ValueLayout { tag });
        m.table.insert(key, value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_get_typed(map: i32, key: i32, tag: i32, _metadata: i32) -> i32 {
    unsafe { IntMap::at(map) }.map_or(0, |m| m.read_typed(m.table.get(&key), tag))
}

/// The value read as `Null<Int>`: the Int, or `NULL_INT` for an absent key.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_get_null_int(map: i32, key: i32, tag: i32, _metadata: i32) -> i64 {
    let Some(m) = (unsafe { IntMap::at(map) }) else {
        return NULL_INT;
    };
    m.table
        .get(&key)
        .map_or(NULL_INT, |&bits| null_int_of(m.layout, bits, tag))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_set_typed(
    map: i32,
    key: i32,
    value: i32,
    tag: i32,
    _metadata: i32,
) {
    if let Some(m) = unsafe { ObjMap::at(map) } {
        let value = m.layout.write(value, ValueLayout { tag });
        m.table.insert(key, value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_get_typed(map: i32, key: i32, tag: i32, _metadata: i32) -> i32 {
    unsafe { ObjMap::at(map) }.map_or(0, |m| m.read_typed(m.table.get(&key), tag))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_count(map: i32) -> i32 {
    unsafe { StrMap::count(map) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_count(map: i32) -> i32 {
    unsafe { IntMap::count(map) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_count(map: i32) -> i32 {
    unsafe { ObjMap::count(map) }
}

/// Keys as a buffer of fresh HaxeString pointers, one per 8-byte slot.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_keys(map: i32, out_len: i32) -> i32 {
    let words = unsafe { StrMap::at(map) }.map_or_else(Vec::new, |m| {
        m.table
            .keys()
            .map(|k| wasm_string_of(k) as u32 as i64)
            .collect()
    });
    unsafe { leak_words(words, out_len) }
}

/// Keys as a buffer of Ints, one per 8-byte slot.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_keys(map: i32, out_len: i32) -> i32 {
    let words = unsafe { IntMap::at(map) }.map_or_else(Vec::new, |m| {
        m.table.keys().map(|&k| i64::from(k)).collect()
    });
    unsafe { leak_words(words, out_len) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_stringmap_to_string_typed(map: i32, val_tag: i32) -> i32 {
    unsafe { StrMap::to_string(map, "null", |k| str_key_text(k), Some(val_tag)) }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_intmap_to_string_typed(map: i32, val_tag: i32) -> i32 {
    unsafe { IntMap::to_string(map, "null", |k| k.to_string(), Some(val_tag)) }
}

/// Keys render the way `"" + key` does for their static type.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_objectmap_to_string_typed(map: i32, key_tag: i32, val_tag: i32) -> i32 {
    let key = |k: &i32| ValueLayout { tag: key_tag }.render_as(*k, key_tag);
    unsafe { ObjMap::to_string(map, "null", key, Some(val_tag)) }
}
