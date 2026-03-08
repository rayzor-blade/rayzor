//! Native JSON parser and stringifier for Rayzor runtime.
//!
//! Provides `haxe_json_parse` and `haxe_json_stringify` as fast native
//! replacements for the pure-Haxe `haxe.format.JsonParser` / `JsonPrinter`.
//!
//! ## Return conventions
//!
//! `haxe_json_parse` returns `*mut u8` → always a heap-allocated `DynamicValue*`:
//! - Objects → `DynamicValue { TYPE_ANON_OBJECT, anon_handle }`
//! - Arrays  → `DynamicValue { TYPE_ARRAY, haxe_array_ptr }`
//! - Strings → `DynamicValue { TYPE_STRING, string_ptr }`
//! - Ints    → `DynamicValue { TYPE_INT, value_ptr }`
//! - Floats  → `DynamicValue { TYPE_FLOAT, value_ptr }`
//! - Bools   → `DynamicValue { TYPE_BOOL, value_ptr }`
//! - null    → null pointer
//!
//! ## Optimizations
//!
//! - Integer accumulation: digits are accumulated into i64 directly during
//!   scanning, avoiding a second pass through `str::parse`.
//! - Zero-copy string keys: object keys without escape sequences are passed
//!   as `&[u8]` slices into the input buffer — no heap allocation.
//! - Byte-level stringify: ASCII bytes are written directly; only non-ASCII
//!   triggers per-char decoding.

use crate::anon_object::{self, DYNAMIC_SHAPE};
use crate::haxe_string::HaxeString;
use crate::type_system::{
    DynamicValue, StringPtr, TypeId, TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_STRING,
};

/// Type ID for arrays in the DynamicValue type system
pub const TYPE_ARRAY: TypeId = TypeId(7);

// Whitespace lookup table (space, tab, newline, carriage return)
const WS: [bool; 256] = {
    let mut t = [false; 256];
    t[b' ' as usize] = true;
    t[b'\t' as usize] = true;
    t[b'\n' as usize] = true;
    t[b'\r' as usize] = true;
    t
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a JSON string and return a DynamicValue*.
///
/// `str_ptr` is a `*const HaxeString` (the compiler passes String as PtrU8).
#[no_mangle]
pub extern "C" fn haxe_json_parse(str_ptr: *const u8) -> *mut u8 {
    if str_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let bytes = unsafe {
        let hs = &*(str_ptr as *const HaxeString);
        if hs.ptr.is_null() || hs.len == 0 {
            return std::ptr::null_mut();
        }
        std::slice::from_raw_parts(hs.ptr, hs.len)
    };

    let mut parser = Parser { bytes, pos: 0 };
    parser.parse_value()
}

/// Stringify a Dynamic value to JSON.
///
/// `value_ptr` is a `*mut u8` pointing to a DynamicValue, an AnonObject handle,
/// or a HaxeArray pointer (depending on how it was produced).
/// Returns a `*mut HaxeString`.
#[no_mangle]
pub extern "C" fn haxe_json_stringify(value_ptr: *mut u8) -> *mut u8 {
    let mut buf = String::with_capacity(128);
    stringify_value(value_ptr, &mut buf);
    alloc_haxe_string(&buf)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    #[inline(always)]
    fn peek(&self) -> Option<u8> {
        if self.pos < self.bytes.len() {
            Some(unsafe { *self.bytes.get_unchecked(self.pos) })
        } else {
            None
        }
    }

    #[inline(always)]
    fn peek_unchecked(&self) -> u8 {
        debug_assert!(self.pos < self.bytes.len());
        unsafe { *self.bytes.get_unchecked(self.pos) }
    }

    #[inline(always)]
    fn advance(&mut self) -> Option<u8> {
        if self.pos < self.bytes.len() {
            let b = unsafe { *self.bytes.get_unchecked(self.pos) };
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }

    #[inline(always)]
    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len()
            && WS[unsafe { *self.bytes.get_unchecked(self.pos) } as usize]
        {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> *mut u8 {
        self.skip_whitespace();
        if self.pos >= self.bytes.len() {
            return std::ptr::null_mut();
        }
        match self.peek_unchecked() {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string_value(),
            b't' => self.parse_true(),
            b'f' => self.parse_false(),
            b'n' => self.parse_null(),
            c if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            _ => std::ptr::null_mut(),
        }
    }

    // -- Object --

    fn parse_object(&mut self) -> *mut u8 {
        self.pos += 1; // skip '{'

        let obj = anon_object::rayzor_anon_new(DYNAMIC_SHAPE, 0);

        self.skip_whitespace();
        if self.pos < self.bytes.len() && self.peek_unchecked() == b'}' {
            self.pos += 1;
            return box_anon(obj);
        }

        loop {
            self.skip_whitespace();
            if self.pos >= self.bytes.len() || self.peek_unchecked() != b'"' {
                break;
            }

            // Parse key — zero-copy when no escapes
            let key = self.parse_string_key();

            self.skip_whitespace();
            if self.advance() != Some(b':') {
                break;
            }

            // Parse value
            let value = self.parse_value();

            // Set field on anon object — key is either a slice into input or owned
            match key {
                StringKey::Borrowed(slice) => {
                    anon_object::rayzor_anon_set_field(
                        obj,
                        slice.as_ptr(),
                        slice.len() as u32,
                        value,
                    );
                }
                StringKey::Owned(ref s) => {
                    anon_object::rayzor_anon_set_field(obj, s.as_ptr(), s.len() as u32, value);
                }
            }

            self.skip_whitespace();
            if self.pos >= self.bytes.len() {
                break;
            }
            match self.peek_unchecked() {
                b',' => {
                    self.pos += 1;
                }
                b'}' => {
                    self.pos += 1;
                    return box_anon(obj);
                }
                _ => break,
            }
        }

        box_anon(obj)
    }

    // -- Array --

    fn parse_array(&mut self) -> *mut u8 {
        self.pos += 1; // skip '['

        let arr = alloc_haxe_array();

        self.skip_whitespace();
        if self.pos < self.bytes.len() && self.peek_unchecked() == b']' {
            self.pos += 1;
            return box_array(arr);
        }

        loop {
            let elem = self.parse_value();
            crate::haxe_array::haxe_array_push_i64(
                arr as *mut crate::haxe_array::HaxeArray,
                elem as i64,
            );

            self.skip_whitespace();
            if self.pos >= self.bytes.len() {
                break;
            }
            match self.peek_unchecked() {
                b',' => {
                    self.pos += 1;
                }
                b']' => {
                    self.pos += 1;
                    return box_array(arr);
                }
                _ => break,
            }
        }

        box_array(arr)
    }

    // -- String --

    /// Parse a JSON string and return it as a boxed DynamicValue* (TYPE_STRING).
    fn parse_string_value(&mut self) -> *mut u8 {
        self.pos += 1; // skip opening '"'
        let start = self.pos;

        // Fast scan: look for closing quote or backslash
        while self.pos < self.bytes.len() {
            let b = unsafe { *self.bytes.get_unchecked(self.pos) };
            if b == b'"' {
                // Fast path: no escapes — zero-copy into StringPtr
                let slice = &self.bytes[start..self.pos];
                self.pos += 1;
                return box_string_from_bytes(slice);
            }
            if b == b'\\' {
                // Switch to slow path
                let prefix = &self.bytes[start..self.pos];
                let s = self.parse_string_slow(prefix);
                return box_string(&s);
            }
            self.pos += 1;
        }

        // Unterminated string — return what we have
        let slice = &self.bytes[start..self.pos];
        box_string_from_bytes(slice)
    }

    /// Parse a JSON object key. Returns a zero-copy slice for unescaped keys.
    fn parse_string_key(&mut self) -> StringKey<'a> {
        self.pos += 1; // skip opening '"'
        let start = self.pos;

        // Fast scan for closing quote
        while self.pos < self.bytes.len() {
            let b = unsafe { *self.bytes.get_unchecked(self.pos) };
            if b == b'"' {
                let slice = &self.bytes[start..self.pos];
                self.pos += 1;
                return StringKey::Borrowed(slice);
            }
            if b == b'\\' {
                let prefix = &self.bytes[start..self.pos];
                let s = self.parse_string_slow(prefix);
                return StringKey::Owned(s);
            }
            self.pos += 1;
        }

        StringKey::Borrowed(&self.bytes[start..self.pos])
    }

    /// Slow path for strings with escape sequences. `prefix` is the already-
    /// scanned bytes before the first backslash.
    fn parse_string_slow(&mut self, prefix: &[u8]) -> String {
        let mut result = String::with_capacity(prefix.len() + 16);
        if let Ok(s) = std::str::from_utf8(prefix) {
            result.push_str(s);
        }

        loop {
            if self.pos >= self.bytes.len() {
                break;
            }
            let b = unsafe { *self.bytes.get_unchecked(self.pos) };
            match b {
                b'"' => {
                    self.pos += 1;
                    return result;
                }
                b'\\' => {
                    self.pos += 1;
                    match self.advance() {
                        Some(b'"') => result.push('"'),
                        Some(b'\\') => result.push('\\'),
                        Some(b'/') => result.push('/'),
                        Some(b'n') => result.push('\n'),
                        Some(b'r') => result.push('\r'),
                        Some(b't') => result.push('\t'),
                        Some(b'b') => result.push('\u{0008}'),
                        Some(b'f') => result.push('\u{000C}'),
                        Some(b'u') => {
                            let cp = self.parse_unicode_escape();
                            if let Some(ch) = char::from_u32(cp) {
                                result.push(ch);
                            }
                        }
                        _ => {}
                    }
                }
                _ => {
                    if b < 0x80 {
                        result.push(b as char);
                        self.pos += 1;
                    } else {
                        let remaining = &self.bytes[self.pos..];
                        if let Some(ch) = std::str::from_utf8(remaining)
                            .ok()
                            .and_then(|s| s.chars().next())
                        {
                            result.push(ch);
                            self.pos += ch.len_utf8();
                        } else {
                            self.pos += 1;
                        }
                    }
                }
            }
        }

        result
    }

    fn parse_unicode_escape(&mut self) -> u32 {
        let mut cp = 0u32;
        for _ in 0..4 {
            if let Some(b) = self.advance() {
                let digit = match b {
                    b'0'..=b'9' => (b - b'0') as u32,
                    b'a'..=b'f' => (b - b'a' + 10) as u32,
                    b'A'..=b'F' => (b - b'A' + 10) as u32,
                    _ => 0,
                };
                cp = cp * 16 + digit;
            }
        }
        // Handle surrogate pairs
        if (0xD800..=0xDBFF).contains(&cp) {
            if self.peek() == Some(b'\\') {
                let saved = self.pos;
                self.pos += 1;
                if self.peek() == Some(b'u') {
                    self.pos += 1;
                    let low = self.parse_unicode_escape();
                    if (0xDC00..=0xDFFF).contains(&low) {
                        return ((cp - 0xD800) << 10) + (low - 0xDC00) + 0x10000;
                    }
                }
                self.pos = saved;
            }
            0xFFFD
        } else {
            cp
        }
    }

    // -- Number --

    fn parse_number(&mut self) -> *mut u8 {
        let start = self.pos;
        let mut is_float = false;
        let mut neg = false;

        // Optional minus
        if self.pos < self.bytes.len() && self.peek_unchecked() == b'-' {
            neg = true;
            self.pos += 1;
        }

        // Integer part — accumulate directly
        let mut int_val: u64 = 0;
        let mut overflow = false;
        while self.pos < self.bytes.len() {
            let b = self.peek_unchecked();
            if b.is_ascii_digit() {
                let d = (b - b'0') as u64;
                // Check overflow before multiply+add
                let (v1, o1) = int_val.overflowing_mul(10);
                let (v2, o2) = v1.overflowing_add(d);
                if o1 || o2 {
                    overflow = true;
                }
                int_val = v2;
                self.pos += 1;
            } else {
                break;
            }
        }

        // Fractional part
        if self.pos < self.bytes.len() && self.peek_unchecked() == b'.' {
            is_float = true;
            self.pos += 1;
            while self.pos < self.bytes.len() && self.peek_unchecked().is_ascii_digit() {
                self.pos += 1;
            }
        }

        // Exponent part
        if self.pos < self.bytes.len() && matches!(self.peek_unchecked(), b'e' | b'E') {
            is_float = true;
            self.pos += 1;
            if self.pos < self.bytes.len() && matches!(self.peek_unchecked(), b'+' | b'-') {
                self.pos += 1;
            }
            while self.pos < self.bytes.len() && self.peek_unchecked().is_ascii_digit() {
                self.pos += 1;
            }
        }

        if is_float || overflow {
            // Fall back to str::parse for floats and overflowing integers
            let num_str = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("0");
            if is_float {
                let f: f64 = num_str.parse().unwrap_or(0.0);
                box_float(f)
            } else {
                // Very large integer — try as float
                let f: f64 = num_str.parse().unwrap_or(0.0);
                box_float(f)
            }
        } else {
            // Integer fast path — value already accumulated
            let i = if neg {
                -(int_val as i64)
            } else {
                int_val as i64
            };
            box_int(i)
        }
    }

    // -- Literals --

    fn parse_true(&mut self) -> *mut u8 {
        if self.bytes.len() - self.pos >= 4
            && unsafe { self.bytes.get_unchecked(self.pos..self.pos + 4) } == b"true"
        {
            self.pos += 4;
            box_bool(true)
        } else {
            std::ptr::null_mut()
        }
    }

    fn parse_false(&mut self) -> *mut u8 {
        if self.bytes.len() - self.pos >= 5
            && unsafe { self.bytes.get_unchecked(self.pos..self.pos + 5) } == b"false"
        {
            self.pos += 5;
            box_bool(false)
        } else {
            std::ptr::null_mut()
        }
    }

    fn parse_null(&mut self) -> *mut u8 {
        if self.bytes.len() - self.pos >= 4
            && unsafe { self.bytes.get_unchecked(self.pos..self.pos + 4) } == b"null"
        {
            self.pos += 4;
            std::ptr::null_mut()
        } else {
            std::ptr::null_mut()
        }
    }
}

/// Result of parsing a JSON string key — either a zero-copy slice or an owned
/// String (when escape sequences are present).
enum StringKey<'a> {
    Borrowed(&'a [u8]),
    Owned(String),
}

// ---------------------------------------------------------------------------
// Boxing helpers — produce heap-allocated DynamicValue*
//
// Must use indirection (Box for inner value) because the rest of the runtime
// dereferences `value_ptr` in `haxe_unbox_*` and `rayzor_anon_set_field`.
// ---------------------------------------------------------------------------

#[inline]
fn box_int(value: i64) -> *mut u8 {
    let val_ptr = Box::into_raw(Box::new(value)) as *mut u8;
    let dv = DynamicValue {
        type_id: TYPE_INT,
        value_ptr: val_ptr,
    };
    Box::into_raw(Box::new(dv)) as *mut u8
}

#[inline]
fn box_float(value: f64) -> *mut u8 {
    let val_ptr = Box::into_raw(Box::new(value)) as *mut u8;
    let dv = DynamicValue {
        type_id: TYPE_FLOAT,
        value_ptr: val_ptr,
    };
    Box::into_raw(Box::new(dv)) as *mut u8
}

#[inline]
fn box_bool(value: bool) -> *mut u8 {
    let val_ptr = Box::into_raw(Box::new(value as i64)) as *mut u8;
    let dv = DynamicValue {
        type_id: TYPE_BOOL,
        value_ptr: val_ptr,
    };
    Box::into_raw(Box::new(dv)) as *mut u8
}

fn box_string(s: &str) -> *mut u8 {
    let bytes = s.as_bytes().to_vec();
    let len = bytes.len();
    let ptr = bytes.as_ptr();
    std::mem::forget(bytes);

    let sp = Box::into_raw(Box::new(StringPtr { ptr, len })) as *mut u8;

    let dv = DynamicValue {
        type_id: TYPE_STRING,
        value_ptr: sp,
    };
    Box::into_raw(Box::new(dv)) as *mut u8
}

/// Box a string from raw bytes — avoids the intermediate Rust String allocation.
fn box_string_from_bytes(bytes: &[u8]) -> *mut u8 {
    let owned = bytes.to_vec();
    let len = owned.len();
    let ptr = owned.as_ptr();
    std::mem::forget(owned);

    let sp = Box::into_raw(Box::new(StringPtr { ptr, len })) as *mut u8;

    let dv = DynamicValue {
        type_id: TYPE_STRING,
        value_ptr: sp,
    };
    Box::into_raw(Box::new(dv)) as *mut u8
}

fn box_anon(anon_handle: *mut u8) -> *mut u8 {
    let dv = DynamicValue {
        type_id: anon_object::TYPE_ANON_OBJECT,
        value_ptr: anon_handle,
    };
    Box::into_raw(Box::new(dv)) as *mut u8
}

fn box_array(arr_ptr: *mut u8) -> *mut u8 {
    let dv = DynamicValue {
        type_id: TYPE_ARRAY,
        value_ptr: arr_ptr,
    };
    Box::into_raw(Box::new(dv)) as *mut u8
}

/// Allocate a zeroed HaxeArray on the heap with elem_size=8.
fn alloc_haxe_array() -> *mut u8 {
    let arr = Box::new(crate::haxe_array::HaxeArray {
        ptr: std::ptr::null_mut(),
        len: 0,
        cap: 0,
        elem_size: 8,
    });
    Box::into_raw(arr) as *mut u8
}

/// Allocate a HaxeString and return it as `*mut u8`.
fn alloc_haxe_string(s: &str) -> *mut u8 {
    let bytes = s.as_bytes().to_vec();
    let len = bytes.len();
    let cap = bytes.capacity();
    let ptr = bytes.as_ptr() as *mut u8;
    std::mem::forget(bytes);
    Box::into_raw(Box::new(HaxeString { ptr, len, cap })) as *mut u8
}

// ---------------------------------------------------------------------------
// Stringify
// ---------------------------------------------------------------------------

fn stringify_value(ptr: *mut u8, buf: &mut String) {
    if ptr.is_null() {
        buf.push_str("null");
        return;
    }

    unsafe {
        let dv = *(ptr as *const DynamicValue);

        if dv.type_id == TYPE_INT {
            if !dv.value_ptr.is_null() {
                let v = *(dv.value_ptr as *const i64);
                itoa_i64(v, buf);
            } else {
                buf.push('0');
            }
        } else if dv.type_id == TYPE_FLOAT {
            if !dv.value_ptr.is_null() {
                let v = *(dv.value_ptr as *const f64);
                if v.fract() == 0.0 && v.is_finite() && v.abs() < 1e15 {
                    itoa_i64(v as i64, buf);
                    buf.push_str(".0");
                } else {
                    buf.push_str(&v.to_string());
                }
            } else {
                buf.push_str("0.0");
            }
        } else if dv.type_id == TYPE_BOOL {
            if !dv.value_ptr.is_null() {
                let v = *(dv.value_ptr as *const i64);
                buf.push_str(if v != 0 { "true" } else { "false" });
            } else {
                buf.push_str("false");
            }
        } else if dv.type_id == TYPE_STRING {
            if !dv.value_ptr.is_null() {
                let sp = &*(dv.value_ptr as *const StringPtr);
                if !sp.ptr.is_null() && sp.len > 0 {
                    let bytes = std::slice::from_raw_parts(sp.ptr, sp.len);
                    stringify_string_bytes(bytes, buf);
                } else {
                    buf.push_str("\"\"");
                }
            } else {
                buf.push_str("\"\"");
            }
        } else if dv.type_id == anon_object::TYPE_ANON_OBJECT {
            stringify_anon_object(dv.value_ptr, buf);
        } else if dv.type_id == TYPE_ARRAY {
            stringify_array(dv.value_ptr, buf);
        } else {
            buf.push_str("null");
        }
    }
}

/// Fast integer-to-string without allocating a temporary String.
fn itoa_i64(v: i64, buf: &mut String) {
    let mut tmp = [0u8; 20]; // i64::MIN has 20 chars
    let mut n = v;
    let negative = n < 0;

    if negative {
        // Handle MIN specially to avoid overflow on negate
        if n == i64::MIN {
            buf.push_str("-9223372036854775808");
            return;
        }
        n = -n;
    }

    let mut i = tmp.len();
    if n == 0 {
        i -= 1;
        tmp[i] = b'0';
    } else {
        while n > 0 {
            i -= 1;
            tmp[i] = b'0' + (n % 10) as u8;
            n /= 10;
        }
    }

    if negative {
        i -= 1;
        tmp[i] = b'-';
    }

    // SAFETY: digits are always valid ASCII
    buf.push_str(unsafe { std::str::from_utf8_unchecked(&tmp[i..]) });
}

/// Byte-level JSON string escaping. For ASCII-heavy content (typical JSON),
/// this avoids the overhead of decoding UTF-8 into chars.
fn stringify_string_bytes(bytes: &[u8], buf: &mut String) {
    buf.push('"');

    let mut start = 0;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        let escape = match b {
            b'"' => Some("\\\""),
            b'\\' => Some("\\\\"),
            b'\n' => Some("\\n"),
            b'\r' => Some("\\r"),
            b'\t' => Some("\\t"),
            0x08 => Some("\\b"),
            0x0C => Some("\\f"),
            0x00..=0x1F => None, // other control chars — need \uXXXX
            _ => {
                i += 1;
                continue; // common case: no escaping needed
            }
        };

        // Flush un-escaped segment
        if start < i {
            // SAFETY: we only get here for ASCII bytes (< 0x80), which are
            // valid UTF-8 boundaries. Multi-byte sequences (>= 0x80) hit the
            // `_ =>` arm and continue without breaking the segment.
            buf.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[start..i]) });
        }

        if let Some(esc) = escape {
            buf.push_str(esc);
        } else {
            // Control char needing \uXXXX
            let hex = [
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX_DIGITS[(b >> 4) as usize],
                HEX_DIGITS[(b & 0xF) as usize],
            ];
            buf.push_str(unsafe { std::str::from_utf8_unchecked(&hex) });
        }

        i += 1;
        start = i;
    }

    // Flush remaining
    if start < bytes.len() {
        buf.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[start..]) });
    }

    buf.push('"');
}

const HEX_DIGITS: [u8; 16] = *b"0123456789abcdef";

fn stringify_anon_object(obj_ptr: *mut u8, buf: &mut String) {
    if obj_ptr.is_null() {
        buf.push_str("null");
        return;
    }

    let fields_arr = anon_object::rayzor_anon_fields(obj_ptr);
    if fields_arr.is_null() {
        buf.push_str("{}");
        return;
    }

    buf.push('{');

    unsafe {
        let arr = &*(fields_arr as *const crate::haxe_array::HaxeArray);
        let mut first = true;

        for i in 0..arr.len {
            let name_hs_ptr = *(arr.ptr.add(i * 8) as *const *mut u8);
            if name_hs_ptr.is_null() {
                continue;
            }
            let name_hs = &*(name_hs_ptr as *const HaxeString);
            if name_hs.ptr.is_null() {
                continue;
            }
            let name_bytes = std::slice::from_raw_parts(name_hs.ptr, name_hs.len);

            let val = anon_object::rayzor_anon_get_field(
                obj_ptr,
                name_bytes.as_ptr(),
                name_bytes.len() as u32,
            );

            if !first {
                buf.push(',');
            }
            first = false;

            stringify_string_bytes(name_bytes, buf);
            buf.push(':');
            stringify_value(val, buf);
        }
    }

    buf.push('}');
}

fn stringify_array(arr_ptr: *mut u8, buf: &mut String) {
    if arr_ptr.is_null() {
        buf.push_str("null");
        return;
    }

    buf.push('[');

    unsafe {
        let arr = &*(arr_ptr as *const crate::haxe_array::HaxeArray);
        for i in 0..arr.len {
            if i > 0 {
                buf.push(',');
            }
            let elem = *(arr.ptr.add(i * 8) as *const *mut u8);
            stringify_value(elem, buf);
        }
    }

    buf.push(']');
}
