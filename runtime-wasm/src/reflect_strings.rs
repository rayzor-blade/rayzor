//! EReg, string, Std, Reflect, enum and Dynamic entry points, ported from
//! the native runtime (`runtime/src/{ereg,haxe_sys,type_system,reflect}.rs`).
//!
//! wasm32 layouts relied on:
//! - Dynamic box: `[tag:u32][payload:u32]`, tags 1=Null 2=Bool 3=Int 4=Float
//!   5=reference (`haxe_box_reference_ptr` drops the class id, so the payload
//!   says what it is).
//! - Class instance: class id in the low word of an i64 header.
//! - Enum cell: `[tag:i32][enum id:i32][param:8 bytes]...`, from the
//!   compiler's `malloc` (`rayzor_obj_malloc`), whose size word sits 8 bytes
//!   before the cell and gives the parameter count.
//! - Closure record: `[code:i32 @0][env:i32 @8]`; code is a table index and
//!   takes the env first.
//! - Int64/I64 slots are i32 here; a Float in one has been truncated.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::type_system::{CONSTRUCTORS, IFACE_IMPLS, TO_STRING, VTABLES, haxe_class_implements};
use crate::{
    haxe_array_new, haxe_array_push_i64, haxe_array_to_string, haxe_box_int_ptr,
    haxe_box_reference_ptr, haxe_string_compare, haxe_string_from_bool, haxe_string_from_int,
    haxe_string_to_string, rayzor_anon_new, rayzor_anon_set_field_by_index, rayzor_ensure_shape,
    rayzor_obj_malloc, rayzor_throw_typed, read_haxe_string, wasm_string_of,
};

const TAG_NULL: u32 = 1;
const TAG_BOOL: u32 = 2;
const TAG_INT: u32 = 3;
const TAG_FLOAT: u32 = 4;
const TAG_REF: u32 = 5;

/// Expected-type ids `Std.is` receives for builtins.
const TYPE_STRING: i32 = 5;
const TYPE_ARRAY: i32 = 7;
const TYPE_DYNAMIC_TOKEN: i32 = (u32::MAX - 2) as i32;

/// Null of an unboxed `Null<Int>`.
const NULL_INT: i64 = 1 << 32;

/// Most parameters an enum cell is read with; a larger count is not a cell.
const MAX_ENUM_PARAMS: u32 = 64;

// ============================================================================
// Memory probes
// ============================================================================

fn mem_end() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        core::arch::wasm32::memory_size::<0>() as u64 * 65536
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        u64::MAX
    }
}

/// `[p, p + n)` lies inside linear memory, above the null page.
fn readable(p: u32, n: u32) -> bool {
    p >= 8 && p as u64 + n as u64 <= mem_end()
}

#[inline]
unsafe fn rd32(p: u32) -> u32 {
    unsafe { core::ptr::read_unaligned(p as *const u32) }
}

#[inline]
unsafe fn wr32(p: u32, v: u32) {
    unsafe { core::ptr::write_unaligned(p as *mut u32, v) }
}

/// `p` as a Dynamic box `(tag, payload)`. A scalar box must point at a
/// payload of its type, which keeps an enum cell with a small tag out.
fn as_box(p: i32) -> Option<(u32, u32)> {
    let a = p as u32;
    if a < 0x1000 || a & 3 != 0 || !readable(a, 8) {
        return None;
    }
    let (tag, payload) = unsafe { (rd32(a), rd32(a + 4)) };
    let ok = match tag {
        TAG_NULL => payload == 0,
        TAG_BOOL | TAG_INT => payload & 3 == 0 && readable(payload, 4),
        TAG_FLOAT => payload & 7 == 0 && readable(payload, 8),
        TAG_REF => true,
        _ => false,
    };
    ok.then_some((tag, payload))
}

fn is_class_id(id: u32) -> bool {
    if id == 0 {
        return false;
    }
    let key = id as i32 as i64;
    let in_vtables = VTABLES
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|m| m.contains_key(&id));
    in_vtables
        || TO_STRING
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.contains_key(&key))
        || CONSTRUCTORS
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.contains_key(&key))
        || IFACE_IMPLS
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.contains_key(&key))
}

/// What a pointer-shaped slot that is not a box holds.
#[derive(Clone, Copy)]
enum Ref {
    Null,
    /// Not an address: an erased scalar, a tag-only enum's discriminant.
    Scalar(i32),
    Class(u32),
    Str,
    Array,
    EnumCell {
        tag: i32,
        enum_id: u32,
    },
    Other,
}

fn describe_ref(q: i32) -> Ref {
    let a = q as u32;
    if a == 0 {
        return Ref::Null;
    }
    if a < 0x1000 || a & 3 != 0 || !readable(a, 8) {
        return Ref::Scalar(q);
    }
    let (w0, w1) = unsafe { (rd32(a), rd32(a + 4)) };
    if w1 == 0 && is_class_id(w0) {
        return Ref::Class(w0);
    }
    if looks_array(a) {
        return Ref::Array;
    }
    if looks_string(a) {
        return Ref::Str;
    }
    if enum_cell_params(a).is_some() {
        return Ref::EnumCell {
            tag: w0 as i32,
            enum_id: w1,
        };
    }
    Ref::Other
}

/// `{data, len, cap}` with a NUL at `data[len]`, as every wasm string has.
fn looks_string(a: u32) -> bool {
    if !readable(a, 12) {
        return false;
    }
    let (d, len, cap) = unsafe { read_haxe_string(a as i32) };
    d != 0
        && len <= cap
        && len < (1 << 28)
        && readable(d, len + 1)
        && unsafe { *((d + len) as *const u8) } == 0
}

/// The 32-byte array header: u32 fields at 8-byte strides, high words zero.
fn looks_array(a: u32) -> bool {
    if !readable(a, 32) {
        return false;
    }
    let w = |i: u32| unsafe { rd32(a + 4 * i) };
    let (d, len, cap, es) = (w(0), w(2), w(4), w(6));
    w(1) == 0
        && w(3) == 0
        && w(5) == 0
        && w(7) == 0
        && (es == 4 || es == 8)
        && len <= cap
        && cap < (1 << 28)
        && ((d == 0 && cap == 0) || (d != 0 && readable(d, cap * es)))
}

/// Parameter count of an enum cell, from its allocation's size word.
fn enum_cell_params(a: u32) -> Option<u32> {
    if a < 0x1000 || a & 3 != 0 || !readable(a - 8, 16) {
        return None;
    }
    let (tag, enum_id, total) = unsafe { (rd32(a) as i32, rd32(a + 4), rd32(a - 8)) };
    if !(0..1024).contains(&tag) || enum_id <= 1024 || total < 16 || (total - 16) % 8 != 0 {
        return None;
    }
    let n = (total - 16) / 8;
    (n <= MAX_ENUM_PARAMS && readable(a, 8 + 8 * n)).then_some(n)
}

/// An interface value's object: the fat pointer's first word, unless it
/// already is the instance.
fn iface_identity(v: i32) -> i32 {
    match describe_ref(v) {
        Ref::Class(_) | Ref::Null | Ref::Scalar(_) => v,
        _ if readable(v as u32, 4) => unsafe { rd32(v as u32) as i32 },
        _ => v,
    }
}

unsafe fn string_bytes<'a>(s: u32) -> &'a [u8] {
    unsafe {
        let (d, len, _) = read_haxe_string(s as i32);
        if d == 0 || len == 0 {
            return &[];
        }
        core::slice::from_raw_parts(d as *const u8, len as usize)
    }
}

fn wasm_str(s: &str) -> i32 {
    wasm_string_of(s.as_bytes())
}

fn ordering(o: core::cmp::Ordering) -> i32 {
    o as i32
}

fn float_cmp(a: f64, b: f64) -> i32 {
    if a < b {
        -1
    } else if a > b {
        1
    } else {
        0
    }
}

// ============================================================================
// EReg
// ============================================================================

struct HaxeEReg {
    regex: Engine,
    global: bool,
    last_input: Option<String>,
    /// Byte spans of each group of the last match; group 0 is the match.
    last_captures: Option<Vec<Option<(usize, usize)>>>,
}

impl HaxeEReg {
    /// The whole match as code-point `(pos, len)`.
    fn matched_span(&self) -> Option<(i32, i32)> {
        let (Some(captures), Some(input)) = (&self.last_captures, &self.last_input) else {
            return None;
        };
        let &(start, end) = captures.first()?.as_ref()?;
        let pos = byte_to_char(input, start);
        Some((pos as i32, (byte_to_char(input, end) - pos) as i32))
    }
}

/// The linear-time engine when the pattern fits it, a backtracking one for
/// lookaround and backreferences.
enum Engine {
    Linear(regex::Regex),
    Backtracking(fancy_regex::Regex),
}

impl Engine {
    fn new(pattern: &str) -> Engine {
        if let Ok(regex) = regex::Regex::new(pattern) {
            return Engine::Linear(regex);
        }
        match fancy_regex::Regex::new(pattern) {
            Ok(regex) => Engine::Backtracking(regex),
            // An invalid pattern never matches.
            Err(_) => Engine::Linear(regex::Regex::new("(?:$^)").unwrap()),
        }
    }

    fn captures(&self, s: &str) -> Option<Vec<Option<(usize, usize)>>> {
        match self {
            Engine::Linear(r) => r.captures(s).map(|c| {
                (0..c.len())
                    .map(|i| c.get(i).map(|m| (m.start(), m.end())))
                    .collect()
            }),
            Engine::Backtracking(r) => r.captures(s).ok().flatten().map(|c| {
                (0..c.len())
                    .map(|i| c.get(i).map(|m| (m.start(), m.end())))
                    .collect()
            }),
        }
    }

    fn find(&self, s: &str) -> Option<(usize, usize)> {
        match self {
            Engine::Linear(r) => r.find(s).map(|m| (m.start(), m.end())),
            Engine::Backtracking(r) => r.find(s).ok().flatten().map(|m| (m.start(), m.end())),
        }
    }

    fn split<'h>(&self, s: &'h str) -> Vec<&'h str> {
        match self {
            Engine::Linear(r) => r.split(s).collect(),
            Engine::Backtracking(r) => r.split(s).filter_map(Result::ok).collect(),
        }
    }

    fn replace(&self, s: &str, by: &str, all: bool) -> String {
        match self {
            Engine::Linear(r) if all => r.replace_all(s, by).into_owned(),
            Engine::Linear(r) => r.replace(s, by).into_owned(),
            Engine::Backtracking(r) => r
                .try_replacen(s, if all { 0 } else { 1 }, by)
                .map(|c| c.into_owned())
                .unwrap_or_else(|_| s.to_string()),
        }
    }
}

/// Haxe flags to `(global, inline prefix)`: "gims" is `(true, "(?ims)")`.
fn parse_flags(flags: &str) -> (bool, String) {
    let mut global = false;
    let mut inline = String::new();
    for ch in flags.chars() {
        match ch {
            'g' => global = true,
            'i' | 'm' | 's' => inline.push(ch),
            _ => {}
        }
    }
    let prefix = if inline.is_empty() {
        String::new()
    } else {
        format!("(?{inline})")
    };
    (global, prefix)
}

fn text_of(s: i32) -> String {
    if s == 0 {
        return String::new();
    }
    unsafe { haxe_string_to_string(s) }.unwrap_or_default()
}

fn byte_to_char(s: &str, offset: usize) -> usize {
    s[..offset.min(s.len())].chars().count()
}

/// Byte offset of code point `n`, clamped to the end.
fn char_to_byte(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)
}

unsafe fn ereg_mut<'a>(ereg: i32) -> Option<&'a mut HaxeEReg> {
    (ereg != 0).then(|| unsafe { &mut *(ereg as u32 as usize as *mut HaxeEReg) })
}

fn record_match(
    ereg: &mut HaxeEReg,
    input: String,
    spans: Option<Vec<Option<(usize, usize)>>>,
) -> i32 {
    let hit = spans.is_some() as i32;
    ereg.last_captures = spans;
    ereg.last_input = Some(input);
    hit
}

/// `new EReg(pattern, opts)`: an opaque handle.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_new(pattern: i32, opts: i32) -> i32 {
    let (global, prefix) = parse_flags(&text_of(opts));
    let ereg = Box::new(HaxeEReg {
        regex: Engine::new(&format!("{prefix}{}", text_of(pattern))),
        global,
        last_input: None,
        last_captures: None,
    });
    Box::into_raw(ereg) as usize as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_match(ereg: i32, s: i32) -> i32 {
    let Some(ereg) = (unsafe { ereg_mut(ereg) }) else {
        return 0;
    };
    let input = text_of(s);
    let spans = ereg.regex.captures(&input);
    record_match(ereg, input, spans)
}

/// Group `n` of the last match: null for a group that did not take part; a
/// group outside the pattern throws "Invalid group".
#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_matched(ereg: i32, n: i32) -> i32 {
    let Some(ereg) = (unsafe { ereg_mut(ereg) }) else {
        return 0;
    };
    if let (Some(captures), Some(input)) = (&ereg.last_captures, &ereg.last_input) {
        let Some(group) = usize::try_from(n).ok().and_then(|i| captures.get(i)) else {
            rayzor_throw_typed(wasm_str("Invalid group"), TYPE_STRING as u32);
            return 0;
        };
        if let Some((start, end)) = group {
            return wasm_str(&input[*start..*end]);
        }
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_matched_left(ereg: i32) -> i32 {
    if let Some(ereg) = unsafe { ereg_mut(ereg) }
        && let (Some(captures), Some(input)) = (&ereg.last_captures, &ereg.last_input)
        && let Some(Some((start, _))) = captures.first()
    {
        return wasm_str(&input[..*start]);
    }
    wasm_str("")
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_matched_right(ereg: i32) -> i32 {
    if let Some(ereg) = unsafe { ereg_mut(ereg) }
        && let (Some(captures), Some(input)) = (&ereg.last_captures, &ereg.last_input)
        && let Some(Some((_, end))) = captures.first()
    {
        return wasm_str(&input[*end..]);
    }
    wasm_str("")
}

/// `matchedPos()` as the anonymous `{len:Int, pos:Int}`: fields sorted, so
/// len is slot 0; the shape id is the native runtime's.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_matched_pos_anon(ereg: i32) -> i32 {
    static INIT: std::sync::Once = std::sync::Once::new();
    const SHAPE_ID: i32 = 1000;
    INIT.call_once(|| rayzor_ensure_shape(SHAPE_ID, wasm_str("len:3,pos:3")));
    let (pos, len) = unsafe { ereg_mut(ereg) }
        .and_then(|e| e.matched_span())
        .unwrap_or((-1, 0));
    let handle = rayzor_anon_new(SHAPE_ID, 2);
    rayzor_anon_set_field_by_index(handle, 0, len as i64);
    rayzor_anon_set_field_by_index(handle, 1, pos as i64);
    handle
}

/// Match within `s` from code point `pos` for `len` code points (`-1` is to
/// the end); positions stay relative to the whole of `s`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_match_sub(ereg: i32, s: i32, pos: i32, len: i32) -> i32 {
    let Some(ereg) = (unsafe { ereg_mut(ereg) }) else {
        return 0;
    };
    let input = text_of(s);
    let start = char_to_byte(&input, pos.max(0) as usize);
    let end = if len < 0 {
        input.len()
    } else {
        start + char_to_byte(&input[start..], len as usize)
    };
    let spans = ereg.regex.captures(&input[start..end]).map(|spans| {
        spans
            .into_iter()
            .map(|g| g.map(|(a, b)| (a + start, b + start)))
            .collect()
    });
    record_match(ereg, input, spans)
}

/// `split(s)`: every match when global, else around the first.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_split(ereg: i32, s: i32) -> i32 {
    let arr = haxe_array_new();
    let Some(ereg) = (unsafe { ereg_mut(ereg) }) else {
        return arr;
    };
    if s == 0 {
        return arr;
    }
    let input = text_of(s);
    let parts: Vec<&str> = if ereg.global {
        ereg.regex.split(&input)
    } else if let Some((start, end)) = ereg.regex.find(&input) {
        vec![&input[..start], &input[end..]]
    } else {
        vec![&input]
    };
    for part in parts {
        haxe_array_push_i64(arr, wasm_str(part));
    }
    arr
}

/// `replace(s, by)`: all matches when global, else the first; `$1`..`$9`
/// and `$$` as the regex crate expands them.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_replace(ereg: i32, s: i32, by: i32) -> i32 {
    let Some(ereg) = (unsafe { ereg_mut(ereg) }) else {
        return wasm_str("");
    };
    if s == 0 {
        return wasm_str("");
    }
    wasm_str(&ereg.regex.replace(&text_of(s), &text_of(by), ereg.global))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_ereg_escape(s: i32) -> i32 {
    wasm_str(&regex::escape(&text_of(s)))
}

// ============================================================================
// String
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn haxe_string_upper(s: i32) -> i32 {
    if s == 0 {
        return wasm_string_of(&[]);
    }
    let bytes = unsafe { string_bytes(s as u32) };
    match core::str::from_utf8(bytes) {
        Ok(text) => wasm_str(&text.to_uppercase()),
        Err(_) => wasm_string_of(bytes),
    }
}

/// Character counts of long strings, keyed by data buffer: `(len, chars)`.
/// The wasm string header has no field to cache the count in, as the native
/// one does; `haxe_string_free` evicts an entry with its buffer.
static CHAR_COUNTS: std::sync::Mutex<Option<std::collections::HashMap<u32, (u32, u32)>>> =
    std::sync::Mutex::new(None);

/// Strings shorter than this are counted on every call, not cached.
const CACHE_CHARS_FROM: usize = 256;

fn count_chars(b: &[u8]) -> u32 {
    if b.is_ascii() {
        return b.len() as u32;
    }
    b.iter().filter(|&&c| (c & 0xC0) != 0x80).count() as u32
}

/// Character count of the string whose buffer is `data`.
fn char_count_of(data: u32, b: &[u8]) -> u32 {
    if b.len() < CACHE_CHARS_FROM {
        return count_chars(b);
    }
    let len = b.len() as u32;
    let mut guard = CHAR_COUNTS.lock().unwrap();
    let map = guard.get_or_insert_with(std::collections::HashMap::new);
    if let Some(&(l, n)) = map.get(&data)
        && l == len
    {
        return n;
    }
    let n = count_chars(b);
    map.insert(data, (len, n));
    n
}

pub(crate) fn forget_char_count(data: u32) {
    if let Some(map) = CHAR_COUNTS.lock().unwrap().as_mut() {
        map.remove(&data);
    }
}

/// `charCodeAt` by code point: the code point, or -1 out of range.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_string_char_code_at_ptr(s: i32, index: i32) -> i32 {
    if s == 0 || index < 0 {
        return -1;
    }
    let data = unsafe { read_haxe_string(s).0 };
    let bytes = unsafe { string_bytes(s as u32) };
    let index = index as usize;
    if char_count_of(data, bytes) as usize == bytes.len() {
        return bytes.get(index).map_or(-1, |&c| c as i32);
    }
    let text = match core::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(e) => unsafe { core::str::from_utf8_unchecked(&bytes[..e.valid_up_to()]) },
    };
    text.chars().nth(index).map_or(-1, |c| c as i32)
}

/// `charCodeAt` as an unboxed `Null<Int>`: `NULL_INT` out of range.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_string_char_code_at_null_int(s: i32, index: i32) -> i64 {
    match haxe_string_char_code_at_ptr(s, index) {
        -1 => NULL_INT,
        code => code as i64,
    }
}

// ============================================================================
// Std
// ============================================================================

/// `Std.int`: toward zero; NaN is 0 and infinities saturate before the
/// narrowing to Int, as the native i64 result is narrowed.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_std_int(x: f64) -> i32 {
    let wide = if x.is_nan() {
        0
    } else if x.is_infinite() {
        if x > 0.0 { i64::MAX } else { i64::MIN }
    } else {
        x.trunc() as i64
    };
    wide as i32
}

/// `Std.isOfType(v, t)`: `expected` is a builtin id (2 Bool, 3 Int, 4 Float,
/// 5 String, 7 Array), a class, interface or enum id, or the Dynamic token.
/// Classes match exactly or through the interface registry; the wasm runtime
/// has no superclass table.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_std_is(v: i32, expected: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    if expected == TYPE_DYNAMIC_TOKEN {
        return 1;
    }
    let reference = match as_box(v) {
        Some((TAG_NULL, _)) => return 0,
        Some((TAG_BOOL, _)) => return (expected == TAG_BOOL as i32) as i32,
        Some((TAG_INT, _)) => {
            return (expected == TAG_INT as i32 || expected == TAG_FLOAT as i32) as i32;
        }
        Some((TAG_FLOAT, payload)) => {
            if expected == TAG_FLOAT as i32 {
                return 1;
            }
            let x = unsafe { *(payload as *const f64) };
            let integral =
                x.is_finite() && x.fract() == 0.0 && x >= i32::MIN as f64 && x <= i32::MAX as f64;
            return (expected == TAG_INT as i32 && integral) as i32;
        }
        Some((_, payload)) => payload as i32,
        None => v,
    };
    let hit = match describe_ref(reference) {
        Ref::Str => expected == TYPE_STRING,
        Ref::Array => expected == TYPE_ARRAY,
        Ref::Class(id) => {
            id == expected as u32 || haxe_class_implements(id as i32 as i64, expected as i64) != 0
        }
        Ref::EnumCell { enum_id, .. } => enum_id == expected as u32,
        // A raw erased scalar outside a box is an Int.
        Ref::Scalar(_) if reference == v => {
            expected == TAG_INT as i32 || expected == TAG_FLOAT as i32
        }
        _ => false,
    };
    hit as i32
}

/// The class's registered `toString`, else its name. The wasm runtime has
/// no class names, so that is the native fallback, `<object>`.
fn class_to_string(id: u32, obj: i32) -> i32 {
    let record = TO_STRING
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(id as i32 as i64)).copied());
    if let Some(record) = record.map(|r| r as u32).filter(|&r| readable(r, 12)) {
        let (code, env) = unsafe { (rd32(record), rd32(record + 8)) };
        if code != 0 {
            let f: extern "C" fn(i32, i32) -> i32 = unsafe { core::mem::transmute(code as usize) };
            return f(env as i32, obj);
        }
    }
    wasm_str("<object>")
}

fn float_string(x: f64) -> i32 {
    wasm_str(&x.to_string())
}

/// Std.string of the referent of a reference box or a raw pointer slot.
fn reference_to_string(q: i32, raw: bool) -> i32 {
    match describe_ref(q) {
        Ref::Null => wasm_str("null"),
        Ref::Scalar(v) => haxe_string_from_int(v),
        Ref::Class(id) => class_to_string(id, q),
        Ref::Str => q,
        Ref::Array => haxe_array_to_string(q),
        // No enum metadata reaches the wasm runtime: the native unregistered form.
        Ref::EnumCell { tag, enum_id } => wasm_str(&format!("<enum {enum_id}::{tag}>")),
        Ref::Other if raw => haxe_string_from_int(q),
        Ref::Other => wasm_str("<object>"),
    }
}

/// `Std.string(Dynamic)`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_std_string_ptr(p: i32) -> i32 {
    if p == 0 {
        return wasm_str("null");
    }
    match as_box(p) {
        Some((TAG_NULL, _)) => wasm_str("null"),
        Some((TAG_BOOL, payload)) => haxe_string_from_bool(unsafe { rd32(payload) } as i32),
        Some((TAG_INT, payload)) => haxe_string_from_int(unsafe { rd32(payload) } as i32),
        Some((TAG_FLOAT, payload)) => float_string(unsafe { *(payload as *const f64) }),
        Some((_, payload)) => reference_to_string(payload as i32, false),
        None => reference_to_string(p, true),
    }
}

/// Stringify a generic slot by its value tag (the native `ValueTag` space:
/// 0 unresolved, 1 Int, 2 Bool, 4 Float, 5 String, 6 Reference, 8 Interface,
/// 9 Function, 10 NullInt, 11 Enum, 12 Dynamic, 13 Array, 14 Int64). Array
/// slot layouts in `metadata` have no wasm counterpart.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_value_to_string_with_metadata(
    value: i32,
    type_tag: i32,
    _metadata: i32,
) -> i32 {
    match type_tag {
        5 if value == 0 => wasm_str("null"),
        5 => value,
        2 => haxe_string_from_bool(value),
        4 => float_string(value as f64),
        13 if value == 0 => wasm_str("null"),
        13 => haxe_array_to_string(value),
        0 | 6 | 11 | 12 => haxe_std_string_ptr(value),
        8 => haxe_std_string_ptr(iface_identity(value)),
        _ => haxe_string_from_int(value),
    }
}

// ============================================================================
// Reflect
// ============================================================================

/// `Reflect.compare` of two Dynamic values: numbers numerically, strings by
/// bytes, bools as 0/1, references by address; mismatched kinds are 0.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_reflect_compare(a: i32, b: i32) -> i32 {
    if a == 0 && b == 0 {
        return 0;
    }
    if a == 0 {
        return -1;
    }
    if b == 0 {
        return 1;
    }
    let (Some((ta, pa)), Some((tb, pb))) = (as_box(a), as_box(b)) else {
        return compare_references(a, b);
    };
    unsafe {
        match (ta, tb) {
            (TAG_INT, TAG_INT) => ordering((rd32(pa) as i32).cmp(&(rd32(pb) as i32))),
            (TAG_FLOAT, TAG_FLOAT) => float_cmp(*(pa as *const f64), *(pb as *const f64)),
            (TAG_INT, TAG_FLOAT) => float_cmp(rd32(pa) as i32 as f64, *(pb as *const f64)),
            (TAG_FLOAT, TAG_INT) => float_cmp(*(pa as *const f64), rd32(pb) as i32 as f64),
            (TAG_BOOL, TAG_BOOL) => (rd32(pa) != 0) as i32 - (rd32(pb) != 0) as i32,
            (TAG_REF, TAG_REF) => compare_references(pa as i32, pb as i32),
            (x, y) if x == y => ordering(pa.cmp(&pb)),
            _ => 0,
        }
    }
}

/// Strings by content, anything else by address.
fn compare_references(a: i32, b: i32) -> i32 {
    if let (Ref::Str, Ref::Str) = (describe_ref(a), describe_ref(b)) {
        return haxe_string_compare(a, b);
    }
    ordering((a as u32).cmp(&(b as u32)))
}

/// A box's payload, or the slot itself.
fn unbox_reference(v: i32) -> i32 {
    match as_box(v) {
        Some((TAG_REF, payload)) => payload as i32,
        Some((TAG_NULL, _)) => 0,
        _ => v,
    }
}

fn closures_equal(a: i32, b: i32) -> bool {
    if a == b {
        return true;
    }
    let (a, b) = (a as u32, b as u32);
    if !readable(a, 12) || !readable(b, 12) {
        return false;
    }
    unsafe { rd32(a) == rd32(b) && rd32(a + 8) == rd32(b + 8) }
}

/// `Reflect.compare` of two erased slots read by one value tag. A Float slot
/// already holds the truncated value on wasm, so it compares as an Int.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_reflect_compare_typed(a: i32, b: i32, type_tag: i32) -> i32 {
    match type_tag {
        1 | 3 | 4 | 10 => ordering(a.cmp(&b)),
        2 => (a != 0) as i32 - (b != 0) as i32,
        5 => {
            let (a, b) = (unbox_reference(a), unbox_reference(b));
            match (a, b) {
                (0, 0) => 0,
                (0, _) => -1,
                (_, 0) => 1,
                _ => haxe_string_compare(a, b),
            }
        }
        6 | 12 => compare_reference_slots(a, b),
        8 => compare_reference_slots(iface_identity(a), iface_identity(b)),
        9 if closures_equal(a, b) => 0,
        _ => ordering((a as u32).cmp(&(b as u32))),
    }
}

/// Two pointer-shaped slots of unknown provenance: boxes of one scalar
/// compare by value, boxes of one object by that object, else by address.
fn compare_reference_slots(a: i32, b: i32) -> i32 {
    if a == b {
        return 0;
    }
    if a == 0 {
        return -1;
    }
    if b == 0 {
        return 1;
    }
    match (as_box(a), as_box(b)) {
        (Some((TAG_REF, pa)), Some((TAG_REF, pb))) => ordering(pa.cmp(&pb)),
        (Some((ta, _)), Some((tb, _))) => {
            let numeric = |t| t == TAG_INT || t == TAG_FLOAT;
            if (numeric(ta) && numeric(tb)) || (ta == tb && ta == TAG_BOOL) {
                haxe_reflect_compare(a, b)
            } else {
                ordering((a as u32).cmp(&(b as u32)))
            }
        }
        _ => ordering((a as u32).cmp(&(b as u32))),
    }
}

/// `Reflect.isEnumValue`: a payload enum's cell, or a reference box holding
/// a non-address (a tag-only enum's discriminant).
#[unsafe(no_mangle)]
pub extern "C" fn haxe_reflect_is_enum_value(v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    let hit = match as_box(v) {
        Some((TAG_REF, payload)) => matches!(
            describe_ref(payload as i32),
            Ref::EnumCell { .. } | Ref::Scalar(_)
        ),
        Some(_) => false,
        None => matches!(describe_ref(v), Ref::EnumCell { .. }),
    };
    hit as i32
}

// ============================================================================
// Enums
// ============================================================================

enum EnumValue {
    Cell(u32),
    Tag(i32),
}

fn is_cell_of(q: i32, type_id: i32) -> bool {
    matches!(describe_ref(q), Ref::EnumCell { enum_id, .. }
        if type_id == 0 || enum_id == type_id as u32)
}

/// Recover a cell or discriminant from a raw slot or a box of either. `type_id`
/// is the enum's id, or 0 when the call site has only a Dynamic.
fn resolve_enum(value: i32, type_id: i32) -> EnumValue {
    if value == 0 {
        return EnumValue::Tag(0);
    }
    // An exact stamp is decisive, so the cell is tried before the box reading.
    if type_id != 0 && is_cell_of(value, type_id) {
        return EnumValue::Cell(value as u32);
    }
    match as_box(value) {
        Some((TAG_REF, payload)) if is_cell_of(payload as i32, type_id) => EnumValue::Cell(payload),
        Some((TAG_REF, payload)) => EnumValue::Tag(payload as i32),
        Some((TAG_INT, payload)) => EnumValue::Tag(unsafe { rd32(payload) } as i32),
        _ if is_cell_of(value, type_id) => EnumValue::Cell(value as u32),
        _ => EnumValue::Tag(value),
    }
}

/// `Type.enumIndex(e)`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_type_enum_index(value: i32, type_id: i32) -> i32 {
    match resolve_enum(value, type_id) {
        EnumValue::Cell(p) => unsafe { rd32(p) as i32 },
        EnumValue::Tag(t) => t,
    }
}

/// A parameter slot as a Dynamic: a box stays as it is, a recognised object
/// is boxed as a reference, anything else as an Int.
fn box_enum_parameter(slot: i32) -> i32 {
    if as_box(slot).is_some() {
        return slot;
    }
    match describe_ref(slot) {
        Ref::Class(_) | Ref::Str | Ref::Array | Ref::EnumCell { .. } => {
            haxe_box_reference_ptr(slot, 0)
        }
        _ => haxe_box_int_ptr(slot),
    }
}

/// `Type.enumParameters(e)`: an `Array<Dynamic>` of the cell's parameters,
/// empty for a tag-only value.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_type_enum_parameters(value: i32, type_id: i32) -> i32 {
    let arr = haxe_array_new();
    if let EnumValue::Cell(p) = resolve_enum(value, type_id) {
        let count = enum_cell_params(p).unwrap_or(0);
        for i in 0..count {
            let slot = unsafe { rd32(p + 8 + 8 * i) } as i32;
            haxe_array_push_i64(arr, box_enum_parameter(slot));
        }
    }
    arr
}

static NULLARY_CELLS: Mutex<Option<HashMap<(i32, i32), i32>>> = Mutex::new(None);

/// The one `[tag][enum id]` cell of a parameterless variant of an enum whose
/// other variants carry payloads, so two evaluations are the same value.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_enum_nullary_cell(type_id: i32, tag: i32) -> i32 {
    let mut cells = NULLARY_CELLS.lock().unwrap();
    *cells
        .get_or_insert_with(HashMap::new)
        .entry((type_id, tag))
        .or_insert_with(|| {
            let cell = rayzor_obj_malloc(8);
            if cell != 0 {
                unsafe {
                    wr32(cell as u32, tag as u32);
                    wr32(cell as u32 + 4, type_id as u32);
                }
            }
            cell
        })
}

// ============================================================================
// Dynamic
// ============================================================================

/// Haxe `==` between two Dynamic slots.
fn dynamic_equals(a: i32, b: i32) -> bool {
    if a == b {
        return true;
    }
    let (da, db) = (as_box(a), as_box(b));
    if a == 0 || b == 0 {
        let null_box = |d: Option<(u32, u32)>| {
            d.is_some_and(|(t, p)| t == TAG_NULL || (t == TAG_REF && p == 0))
        };
        return null_box(da) || null_box(db);
    }
    let (Some((ta, pa)), Some((tb, pb))) = (da, db) else {
        // A reference can be raw on one side and boxed on the other.
        return unbox_reference(a) == unbox_reference(b);
    };
    unsafe {
        let num = |t: u32, p: u32| match t {
            TAG_INT => Some(rd32(p) as i32 as f64),
            TAG_FLOAT => Some(*(p as *const f64)),
            _ => None,
        };
        if let (Some(x), Some(y)) = (num(ta, pa), num(tb, pb)) {
            return x == y;
        }
        match (ta, tb) {
            (TAG_NULL, TAG_NULL) => true,
            (TAG_BOOL, TAG_BOOL) => (rd32(pa) != 0) == (rd32(pb) != 0),
            (TAG_REF, TAG_REF) => {
                pa == pb
                    || matches!((describe_ref(pa as i32), describe_ref(pb as i32)), (Ref::Str, Ref::Str)
                        if string_bytes(pa) == string_bytes(pb))
            }
            _ => false,
        }
    }
}

/// `==` between a tag-described erased slot and a Dynamic operand; each side
/// is read the way its own provenance says.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_dynamic_equals_typed(raw: i32, type_tag: i32, other: i32) -> i32 {
    let eq = match type_tag {
        8 => {
            let object = iface_identity(raw);
            object == unbox_reference(other) || dynamic_equals(object, other)
        }
        0 | 6 | 12 => dynamic_equals(raw, other),
        _ => match as_box(other) {
            None => raw == other,
            Some((TAG_NULL, _)) => raw == 0,
            Some((tag, payload)) => unsafe {
                match type_tag {
                    1 | 3 | 4 | 10 | 11 => match tag {
                        TAG_INT => rd32(payload) as i32 == raw,
                        TAG_FLOAT => *(payload as *const f64) == raw as f64,
                        TAG_REF => payload as i32 == raw,
                        _ => false,
                    },
                    2 => tag == TAG_BOOL && (rd32(payload) != 0) == (raw != 0),
                    5 => {
                        tag == TAG_REF
                            && if raw == 0 || payload == 0 {
                                raw == payload as i32
                            } else {
                                haxe_string_compare(raw, payload as i32) == 0
                            }
                    }
                    _ => raw == other,
                }
            },
        },
    };
    eq as i32
}

/// A box's payload as an erased slot: an Int or Bool its value, a Float its
/// truncation (wasm slots are i32), a reference its pointer; a non-box
/// passes through.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_dynamic_to_slot(p: i32) -> i32 {
    if p == 0 {
        return 0;
    }
    match as_box(p) {
        Some((TAG_NULL, _)) => 0,
        Some((TAG_INT | TAG_BOOL, payload)) => unsafe { rd32(payload) as i32 },
        Some((TAG_FLOAT, payload)) => unsafe { *(payload as *const f64) as i32 },
        Some((_, payload)) => payload as i32,
        None => p,
    }
}
