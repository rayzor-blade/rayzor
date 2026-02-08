//! EReg (Regular Expression) runtime support
//!
//! Implements the Haxe EReg API backed by Rust's `regex` crate.
//! EReg is an opaque pointer type: Box<HaxeEReg> cast to *mut u8.

use regex::Regex;
use std::alloc::{alloc, Layout};
use std::ptr;

use crate::haxe_array::HaxeArray;
use crate::haxe_string::{haxe_string_from_bytes, HaxeString};

// ============================================================================
// Internal types
// ============================================================================

struct HaxeEReg {
    regex: Regex,
    global: bool,
    /// Last matched input string (cloned from match() call)
    last_input: Option<String>,
    /// Byte offset ranges for each capture group. Index 0 = full match.
    last_captures: Option<Vec<Option<(usize, usize)>>>,
}

/// Convert HaxeString pointer to Rust &str
unsafe fn hs_to_str<'a>(s: *const HaxeString) -> &'a str {
    if s.is_null() || (*s).ptr.is_null() || (*s).len == 0 {
        return "";
    }
    let bytes = std::slice::from_raw_parts((*s).ptr, (*s).len);
    std::str::from_utf8_unchecked(bytes)
}

/// Create a new heap-allocated HaxeString from a Rust &str, return as *mut u8 (i64-sized pointer)
fn rust_str_to_hs(s: &str) -> *mut u8 {
    let hs = Box::new(HaxeString {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    });
    let hs_ptr = Box::into_raw(hs);
    haxe_string_from_bytes(hs_ptr, s.as_ptr(), s.len());
    hs_ptr as *mut u8
}

/// Parse Haxe regex flags string into (global, inline_prefix)
/// e.g. "gims" → (true, "(?ims)")
fn parse_flags(flags_str: &str) -> (bool, String) {
    let mut global = false;
    let mut inline_flags = String::new();
    for ch in flags_str.chars() {
        match ch {
            'g' => global = true,
            'i' | 'm' | 's' => inline_flags.push(ch),
            _ => {} // ignore unknown flags
        }
    }
    let prefix = if inline_flags.is_empty() {
        String::new()
    } else {
        format!("(?{})", inline_flags)
    };
    (global, prefix)
}

// ============================================================================
// Extern C functions
// ============================================================================

/// Create a new EReg from pattern and flags strings.
/// Returns opaque pointer (Box<HaxeEReg> as *mut u8).
#[no_mangle]
pub extern "C" fn haxe_ereg_new(
    pattern: *const HaxeString,
    opts: *const HaxeString,
) -> *mut u8 {
    unsafe {
        let pattern_str = hs_to_str(pattern);
        let opts_str = hs_to_str(opts);
        let (global, prefix) = parse_flags(opts_str);

        let full_pattern = format!("{}{}", prefix, pattern_str);

        let regex = match Regex::new(&full_pattern) {
            Ok(r) => r,
            Err(_) => {
                // On invalid regex, create one that never matches
                Regex::new("(?!.)").unwrap()
            }
        };

        let ereg = Box::new(HaxeEReg {
            regex,
            global,
            last_input: None,
            last_captures: None,
        });
        Box::into_raw(ereg) as *mut u8
    }
}

/// Test if regex matches the string. Updates internal match state.
/// Returns 1 if match found, 0 otherwise.
#[no_mangle]
pub extern "C" fn haxe_ereg_match(ereg: *mut u8, s: *const HaxeString) -> i32 {
    if ereg.is_null() {
        return 0;
    }
    unsafe {
        let ereg = &mut *(ereg as *mut HaxeEReg);
        let input = hs_to_str(s).to_string();

        if let Some(caps) = ereg.regex.captures(&input) {
            let mut capture_ranges = Vec::new();
            for i in 0..caps.len() {
                capture_ranges.push(caps.get(i).map(|m| (m.start(), m.end())));
            }
            ereg.last_captures = Some(capture_ranges);
            ereg.last_input = Some(input);
            1
        } else {
            ereg.last_captures = None;
            ereg.last_input = Some(input);
            0
        }
    }
}

/// Get the nth matched group (0 = full match).
/// Returns HaxeString pointer, or null if no match or group out of range.
#[no_mangle]
pub extern "C" fn haxe_ereg_matched(ereg: *mut u8, n: i32) -> *mut u8 {
    if ereg.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        let ereg = &*(ereg as *mut HaxeEReg);
        if let (Some(captures), Some(input)) = (&ereg.last_captures, &ereg.last_input) {
            let idx = n as usize;
            if idx < captures.len() {
                if let Some((start, end)) = captures[idx] {
                    return rust_str_to_hs(&input[start..end]);
                }
            }
        }
        ptr::null_mut()
    }
}

/// Get the substring before the match.
#[no_mangle]
pub extern "C" fn haxe_ereg_matched_left(ereg: *mut u8) -> *mut u8 {
    if ereg.is_null() {
        return rust_str_to_hs("");
    }
    unsafe {
        let ereg = &*(ereg as *mut HaxeEReg);
        if let (Some(captures), Some(input)) = (&ereg.last_captures, &ereg.last_input) {
            if let Some(Some((start, _end))) = captures.first() {
                return rust_str_to_hs(&input[..*start]);
            }
        }
        rust_str_to_hs("")
    }
}

/// Get the substring after the match.
#[no_mangle]
pub extern "C" fn haxe_ereg_matched_right(ereg: *mut u8) -> *mut u8 {
    if ereg.is_null() {
        return rust_str_to_hs("");
    }
    unsafe {
        let ereg = &*(ereg as *mut HaxeEReg);
        if let (Some(captures), Some(input)) = (&ereg.last_captures, &ereg.last_input) {
            if let Some(Some((_start, end))) = captures.first() {
                return rust_str_to_hs(&input[*end..]);
            }
        }
        rust_str_to_hs("")
    }
}

/// Write match position and length to out-params.
#[no_mangle]
pub extern "C" fn haxe_ereg_matched_pos(
    ereg: *mut u8,
    out_pos: *mut i32,
    out_len: *mut i32,
) {
    if ereg.is_null() || out_pos.is_null() || out_len.is_null() {
        return;
    }
    unsafe {
        let ereg = &*(ereg as *mut HaxeEReg);
        if let Some(captures) = &ereg.last_captures {
            if let Some(Some((start, end))) = captures.first() {
                *out_pos = *start as i32;
                *out_len = (*end - *start) as i32;
                return;
            }
        }
        *out_pos = -1;
        *out_len = 0;
    }
}

/// Match a substring of s starting at pos with optional length.
/// len = -1 means match to end of string.
/// Returns 1 if match found, 0 otherwise.
#[no_mangle]
pub extern "C" fn haxe_ereg_match_sub(
    ereg: *mut u8,
    s: *const HaxeString,
    pos: i32,
    len: i32,
) -> i32 {
    if ereg.is_null() {
        return 0;
    }
    unsafe {
        let ereg_ref = &mut *(ereg as *mut HaxeEReg);
        let full_input = hs_to_str(s);
        let start = (pos as usize).min(full_input.len());
        let end = if len < 0 {
            full_input.len()
        } else {
            (start + len as usize).min(full_input.len())
        };
        let sub = &full_input[start..end];

        if let Some(caps) = ereg_ref.regex.captures(sub) {
            let mut capture_ranges = Vec::new();
            for i in 0..caps.len() {
                // Adjust offsets to be relative to the full input string
                capture_ranges.push(
                    caps.get(i).map(|m| (m.start() + start, m.end() + start)),
                );
            }
            ereg_ref.last_captures = Some(capture_ranges);
            ereg_ref.last_input = Some(full_input.to_string());
            1
        } else {
            ereg_ref.last_captures = None;
            ereg_ref.last_input = Some(full_input.to_string());
            0
        }
    }
}

/// Split string by regex. Returns a HaxeArray of HaxeString pointers.
#[no_mangle]
pub extern "C" fn haxe_ereg_split(
    ereg: *mut u8,
    s: *const HaxeString,
) -> *mut HaxeArray {
    if ereg.is_null() || s.is_null() {
        let arr = Box::new(HaxeArray {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
            elem_size: 8,
        });
        return Box::into_raw(arr);
    }
    unsafe {
        let ereg = &*(ereg as *mut HaxeEReg);
        let input = hs_to_str(s);

        let parts: Vec<&str> = if ereg.global {
            ereg.regex.split(input).collect()
        } else {
            // Non-global: split at first match only → [before, after]
            if let Some(m) = ereg.regex.find(input) {
                vec![&input[..m.start()], &input[m.end()..]]
            } else {
                vec![input]
            }
        };

        let count = parts.len();
        let elem_size = 8usize;
        let total_size = count * elem_size;

        let data_ptr = if total_size > 0 {
            let layout = Layout::from_size_align_unchecked(total_size, 8);
            alloc(layout)
        } else {
            ptr::null_mut()
        };

        if total_size > 0 && data_ptr.is_null() {
            panic!("Failed to allocate memory for ereg split array");
        }

        let i64_ptr = data_ptr as *mut i64;
        for (i, part) in parts.iter().enumerate() {
            let str_ptr = rust_str_to_hs(part);
            *i64_ptr.add(i) = str_ptr as i64;
        }

        let arr = Box::new(HaxeArray {
            ptr: data_ptr,
            len: count,
            cap: count,
            elem_size: 8,
        });
        Box::into_raw(arr)
    }
}

/// Replace matches in the string.
/// If global flag is set, replaces all occurrences; otherwise only the first.
/// Supports $1..$9 backreferences and $$ for literal $.
#[no_mangle]
pub extern "C" fn haxe_ereg_replace(
    ereg: *mut u8,
    s: *const HaxeString,
    by: *const HaxeString,
) -> *mut u8 {
    if ereg.is_null() || s.is_null() {
        return rust_str_to_hs("");
    }
    unsafe {
        let ereg = &*(ereg as *mut HaxeEReg);
        let input = hs_to_str(s);
        let replacement = hs_to_str(by);

        let result = if ereg.global {
            ereg.regex.replace_all(input, replacement).into_owned()
        } else {
            ereg.regex.replace(input, replacement).into_owned()
        };

        rust_str_to_hs(&result)
    }
}

/// Escape regex metacharacters in the string.
/// Static method — does not use an EReg instance.
#[no_mangle]
pub extern "C" fn haxe_ereg_escape(s: *const HaxeString) -> *mut u8 {
    if s.is_null() {
        return rust_str_to_hs("");
    }
    unsafe {
        let input = hs_to_str(s);
        let escaped = regex::escape(input);
        rust_str_to_hs(&escaped)
    }
}
