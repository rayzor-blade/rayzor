//! Haxe String runtime implementation
//!
//! Memory layout: [length: usize, capacity: usize, data...]
//! All strings are UTF-8 encoded and null-terminated for C interop

use log::debug;
use std::alloc::{alloc, dealloc, Layout};
use std::io::Write;
use std::ptr;
use std::slice;
use std::str;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Haxe String representation (pointer-based, no struct returns)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct HaxeString {
    pub ptr: *mut u8, // Pointer to string data (UTF-8)
    pub len: usize,   // Length in bytes
    pub cap: usize,   // Capacity in bytes
}

const INITIAL_CAPACITY: usize = 32;

fn stdout_flush_interval() -> Duration {
    static INTERVAL: OnceLock<Duration> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        let ms = std::env::var("RAYZOR_STDOUT_FLUSH_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(50);
        Duration::from_millis(ms)
    })
}

fn write_stdout_buffered(bytes: &[u8], force: bool) {
    write_stdout_parts(bytes, b"", force)
}

/// The one stdout path. Everything that writes to stdout has to come through
/// here, or it races the buffer: a `print!` goes straight to the fd while a
/// buffered line is still queued, and the two arrive interleaved.
pub fn rayzor_stdout_write(head: &[u8], tail: &[u8], force: bool) {
    write_stdout_parts(head, tail, force)
}

/// Append both parts under ONE acquisition of the buffer lock.
///
/// A line and its newline have to arrive together. Written as two calls, the
/// lock is released between them, and a second thread printing at that moment
/// appends its own line in the gap — the two lines merge into one, so output
/// that was never lost reads as output that went missing.
fn write_stdout_parts(head: &[u8], tail: &[u8], force: bool) {
    static BUFFER: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();
    static LAST_FLUSH: OnceLock<Mutex<Instant>> = OnceLock::new();
    let interval = stdout_flush_interval();

    let Ok(mut buffer) = BUFFER
        .get_or_init(|| Mutex::new(Vec::with_capacity(4096)))
        .lock()
    else {
        return;
    };
    buffer.extend_from_slice(head);
    buffer.extend_from_slice(tail);

    let now = Instant::now();
    let mut should_flush = force || interval.is_zero() || buffer.len() >= 4096;
    if !should_flush {
        if let Ok(last) = LAST_FLUSH.get_or_init(|| Mutex::new(now)).lock() {
            should_flush = now.duration_since(*last) >= interval;
        }
    }
    if !should_flush {
        return;
    }

    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(&buffer);
    let _ = stdout.flush();
    buffer.clear();
    if let Ok(mut last) = LAST_FLUSH.get_or_init(|| Mutex::new(now)).lock() {
        *last = now;
    }
}

// ============================================================================
// String Creation
// ============================================================================

/// Create a new empty string
#[no_mangle]
pub extern "C" fn haxe_string_new(out: *mut HaxeString) {
    crate::panic_guard::guarded_call(|| unsafe {
        let layout = Layout::from_size_align_unchecked(INITIAL_CAPACITY, 1);
        let ptr = alloc(layout);
        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        *ptr = 0; // Null terminator

        (*out).ptr = ptr;
        (*out).len = 0;
        (*out).cap = INITIAL_CAPACITY;
    })
}

/// Create a string from a C string (null-terminated)
#[no_mangle]
pub extern "C" fn haxe_string_from_cstr(out: *mut HaxeString, cstr: *const u8) {
    if cstr.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        // Find length
        let mut len = 0;
        while *cstr.add(len) != 0 {
            len += 1;
        }

        let cap = len.max(INITIAL_CAPACITY) + 1; // +1 for null terminator
        let layout = Layout::from_size_align_unchecked(cap, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        // Copy data
        ptr::copy_nonoverlapping(cstr, ptr, len);
        *ptr.add(len) = 0; // Null terminator

        (*out).ptr = ptr;
        (*out).len = len;
        (*out).cap = cap;
    }
}

/// Create a string from bytes with known length
#[no_mangle]
pub extern "C" fn haxe_string_from_bytes(out: *mut HaxeString, bytes: *const u8, len: usize) {
    if bytes.is_null() || len == 0 {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let cap = len.max(INITIAL_CAPACITY) + 1;
        let layout = Layout::from_size_align_unchecked(cap, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        ptr::copy_nonoverlapping(bytes, ptr, len);
        *ptr.add(len) = 0; // Null terminator

        (*out).ptr = ptr;
        (*out).len = len;
        (*out).cap = cap;
    }
}

// ============================================================================
// String Properties
// ============================================================================

/// Get string length
#[no_mangle]
pub extern "C" fn haxe_string_length(s: *const HaxeString) -> usize {
    if s.is_null() {
        return 0;
    }
    unsafe { (*s).len }
}

/// Get character at index
#[no_mangle]
pub extern "C" fn haxe_string_char_at(s: *const HaxeString, index: usize) -> i32 {
    if s.is_null() {
        return -1;
    }

    unsafe {
        let s_ref = &*s;
        if index >= s_ref.len {
            return -1;
        }
        *s_ref.ptr.add(index) as i32
    }
}

/// Get character code at index
#[no_mangle]
pub extern "C" fn haxe_string_char_code_at(s: *const HaxeString, index: usize) -> i32 {
    haxe_string_char_at(s, index)
}

// ============================================================================
// String Operations
// ============================================================================

/// Concatenate two strings (sret variant — use haxe_string_concat_ptr instead)
#[no_mangle]
pub extern "C" fn haxe_string_concat_sret(
    out: *mut HaxeString,
    a: *const HaxeString,
    b: *const HaxeString,
) {
    if a.is_null() && b.is_null() {
        haxe_string_new(out);
        return;
    }

    crate::panic_guard::guarded_call(|| unsafe {
        let a_len = if a.is_null() { 0 } else { (*a).len };
        let b_len = if b.is_null() { 0 } else { (*b).len };
        let total_len = a_len + b_len;

        let cap = total_len.max(INITIAL_CAPACITY) + 1;
        let layout = Layout::from_size_align_unchecked(cap, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        // Copy first string
        if a_len > 0 {
            ptr::copy_nonoverlapping((*a).ptr, ptr, a_len);
        }

        // Copy second string
        if b_len > 0 {
            ptr::copy_nonoverlapping((*b).ptr, ptr.add(a_len), b_len);
        }

        *ptr.add(total_len) = 0; // Null terminator

        (*out).ptr = ptr;
        (*out).len = total_len;
        (*out).cap = cap;
    })
}

/// Get substring
#[no_mangle]
pub extern "C" fn haxe_string_substring(
    out: *mut HaxeString,
    s: *const HaxeString,
    start: usize,
    end: usize,
) {
    if s.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let s_ref = &*s;
        let actual_start = start.min(s_ref.len);
        let actual_end = end.min(s_ref.len);

        if actual_start >= actual_end {
            haxe_string_new(out);
            return;
        }

        let len = actual_end - actual_start;
        let cap = len.max(INITIAL_CAPACITY) + 1;
        let layout = Layout::from_size_align_unchecked(cap, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        ptr::copy_nonoverlapping(s_ref.ptr.add(actual_start), ptr, len);
        *ptr.add(len) = 0;

        (*out).ptr = ptr;
        (*out).len = len;
        (*out).cap = cap;
    }
}

/// Substring with just start position (to end of string)
#[no_mangle]
pub extern "C" fn haxe_string_substr(
    out: *mut HaxeString,
    s: *const HaxeString,
    start: usize,
    length: usize,
) {
    if s.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let s_ref = &*s;
        let actual_start = start.min(s_ref.len);
        let actual_end = (start + length).min(s_ref.len);
        haxe_string_substring(out, s, actual_start, actual_end);
    }
}

/// Convert to uppercase
#[no_mangle]
pub extern "C" fn haxe_string_to_upper_case(out: *mut HaxeString, s: *const HaxeString) {
    if s.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let s_ref = &*s;
        if s_ref.len == 0 {
            haxe_string_new(out);
            return;
        }

        let slice = slice::from_raw_parts(s_ref.ptr, s_ref.len);
        if let Ok(rust_str) = str::from_utf8(slice) {
            let upper = rust_str.to_uppercase();
            haxe_string_from_bytes(out, upper.as_ptr(), upper.len());
        } else {
            // Invalid UTF-8, just copy
            haxe_string_from_bytes(out, s_ref.ptr, s_ref.len);
        }
    }
}

/// Convert to lowercase
#[no_mangle]
pub extern "C" fn haxe_string_to_lower_case(out: *mut HaxeString, s: *const HaxeString) {
    if s.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let s_ref = &*s;
        if s_ref.len == 0 {
            haxe_string_new(out);
            return;
        }

        let slice = slice::from_raw_parts(s_ref.ptr, s_ref.len);
        if let Ok(rust_str) = str::from_utf8(slice) {
            let lower = rust_str.to_lowercase();
            haxe_string_from_bytes(out, lower.as_ptr(), lower.len());
        } else {
            // Invalid UTF-8, just copy
            haxe_string_from_bytes(out, s_ref.ptr, s_ref.len);
        }
    }
}

/// Index of substring
#[no_mangle]
pub extern "C" fn haxe_string_index_of(
    s: *const HaxeString,
    needle: *const HaxeString,
    start: usize,
) -> i32 {
    if s.is_null() || needle.is_null() {
        return -1;
    }

    unsafe {
        let s_ref = &*s;
        let needle_ref = &*needle;

        if needle_ref.len == 0 || start >= s_ref.len {
            return -1;
        }

        let haystack = slice::from_raw_parts(s_ref.ptr, s_ref.len);
        let needle_bytes = slice::from_raw_parts(needle_ref.ptr, needle_ref.len);

        // Simple substring search
        for i in start..=(s_ref.len.saturating_sub(needle_ref.len)) {
            if &haystack[i..i + needle_ref.len] == needle_bytes {
                return i as i32;
            }
        }

        -1
    }
}

/// Compare two strings lexicographically, returns -1/0/1
#[no_mangle]
pub extern "C" fn haxe_string_compare(a: *const HaxeString, b: *const HaxeString) -> i32 {
    if a.is_null() && b.is_null() {
        return 0;
    }
    if a.is_null() {
        return -1;
    }
    if b.is_null() {
        return 1;
    }
    unsafe {
        let a_ref = &*a;
        let b_ref = &*b;
        let a_bytes = slice::from_raw_parts(a_ref.ptr, a_ref.len);
        let b_bytes = slice::from_raw_parts(b_ref.ptr, b_ref.len);
        match a_bytes.cmp(b_bytes) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

/// Split string by delimiter
#[no_mangle]
pub extern "C" fn haxe_string_split(
    out: *mut *mut HaxeString,
    out_len: *mut usize,
    s: *const HaxeString,
    delimiter: *const HaxeString,
) {
    debug!("[OLD haxe_string_split] Called!");
    if s.is_null() || delimiter.is_null() {
        unsafe {
            *out = ptr::null_mut();
            *out_len = 0;
        }
        return;
    }

    unsafe {
        let s_ref = &*s;
        let delim_ref = &*delimiter;

        debug!(
            "[OLD split] s.len={}, delimiter.len={}",
            s_ref.len, delim_ref.len
        );

        // Count occurrences
        let mut count = 1;
        let mut pos = 0;
        loop {
            let idx = haxe_string_index_of(s, delimiter, pos);
            if idx < 0 {
                break;
            }
            count += 1;
            pos = (idx as usize) + delim_ref.len;
        }

        // Allocate array of HaxeString
        let layout = Layout::array::<HaxeString>(count).unwrap();
        let array_ptr = alloc(layout) as *mut HaxeString;

        // Fill array
        let mut array_idx = 0;
        let mut start = 0;
        loop {
            let idx = haxe_string_index_of(s, delimiter, start);
            if idx < 0 {
                // Last part
                haxe_string_substring(array_ptr.add(array_idx), s, start, s_ref.len);
                break;
            }

            haxe_string_substring(array_ptr.add(array_idx), s, start, idx as usize);
            array_idx += 1;
            start = (idx as usize) + delim_ref.len;
        }

        *out = array_ptr;
        *out_len = count;
    }
}

/// Split string into an array of strings (returns proper HaxeArray)
/// This is the preferred version that returns Array<String> properly
#[no_mangle]
pub extern "C" fn haxe_string_split_array(
    s: *const HaxeString,
    delimiter: *const HaxeString,
) -> *mut crate::haxe_array::HaxeArray {
    use crate::haxe_array::HaxeArray;

    debug!(
        "[split] Function entry: s={:?}, delimiter={:?}",
        s, delimiter
    );

    if s.is_null() || delimiter.is_null() {
        // Return empty array
        let arr = Box::new(HaxeArray {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
            elem_size: 8, // size of pointer (i64)
        });
        return Box::into_raw(arr);
    }

    unsafe {
        let s_ref = &*s;
        let delim_ref = &*delimiter;

        debug!(
            "[split] s.len={}, delimiter.len={}",
            s_ref.len, delim_ref.len
        );

        // Count occurrences
        let mut count = 1;
        let mut pos = 0;
        loop {
            let idx = haxe_string_index_of(s, delimiter, pos);
            debug!("[split] index_of from pos={} returned idx={}", pos, idx);
            if idx < 0 {
                break;
            }
            count += 1;
            pos = (idx as usize) + delim_ref.len;
        }
        debug!("[split] Final count={}", count);

        // Create HaxeArray to hold string pointers as i64
        let elem_size = 8; // size of pointer
        let total_size = count * elem_size;
        let layout = Layout::from_size_align_unchecked(total_size, 8);
        let data_ptr = alloc(layout);

        if data_ptr.is_null() {
            panic!("Failed to allocate memory for string split array");
        }

        // Fill array with string pointers
        let mut array_idx = 0;
        let mut start = 0;
        let i64_ptr = data_ptr as *mut i64;

        loop {
            let idx = haxe_string_index_of(s, delimiter, start);
            if idx < 0 {
                // Last part - allocate and store substring
                let substring = Box::new(HaxeString {
                    ptr: ptr::null_mut(),
                    len: 0,
                    cap: 0,
                });
                let substr_ptr = Box::into_raw(substring);
                haxe_string_substring(substr_ptr, s, start, s_ref.len);
                *i64_ptr.add(array_idx) = substr_ptr as i64;
                break;
            }

            // Allocate and store substring
            let substring = Box::new(HaxeString {
                ptr: ptr::null_mut(),
                len: 0,
                cap: 0,
            });
            let substr_ptr = Box::into_raw(substring);
            haxe_string_substring(substr_ptr, s, start, idx as usize);
            *i64_ptr.add(array_idx) = substr_ptr as i64;

            array_idx += 1;
            start = (idx as usize) + delim_ref.len;
        }

        // Create and return HaxeArray
        let arr = Box::new(HaxeArray {
            ptr: data_ptr,
            len: count,
            cap: count,
            elem_size: 8,
        });
        let arr_ptr = Box::into_raw(arr);
        debug!(
            "[split] Returning HaxeArray pointer: {:?} (count={})",
            arr_ptr, count
        );
        arr_ptr
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free string memory.
///
/// Every `*mut HaxeString` that flows through this sink is produced by a
/// `Box::into_raw(Box::new(HaxeString { .. }))` constructor (native_stack_trace,
/// haxe_sys, the `replace` / `split` substrings below, and the rest of the
/// runtime). The 24-byte HaxeString header itself is therefore heap-owned and
/// must be reclaimed via `Box::from_raw` to match Rust's GlobalAlloc Layout
/// (size 24, align 8) — calling `dealloc` with the buffer Layout (align 1)
/// would mis-deallocate the header, and skipping the reclaim leaks 24 B per
/// string AND keeps a dangling `ptr` field alive in the struct (the UAF /
/// double-free window the test_exception flake hit on ~25 % of runs).
///
/// We first capture the byte-buffer fields, zero them in place so any spurious
/// second free short-circuits on the buffer path, and only then reclaim the
/// header. The drop order matters: the `Box::from_raw` runs after the buffer
/// dealloc so we never read through a freed header.
/// Env-gated (`RZT_DBG_STRFREE=1`) call counter — proves the InsertFree
/// string-release path fires at runtime (mirrors haxe_array_free's counter).
/// Whether to keep released headers so a second release can be recognised.
/// Read once; the diagnostic leaks, so it is opt-in.
fn strfree_keep_headers() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RZT_DBG_STRFREE").is_some())
}

/// Written into a released header's length so a second release can recognise
/// it — reported under `RZT_DBG_STRFREE`. Only readable while the freed block
/// is still untouched, so its absence proves nothing.
const FREED_MARK: usize = 0xF4EE_D0F4_EED0;

fn strfree_dbg_count() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("RZT_DBG_STRFREE").is_some()) {
        return;
    }
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n.is_multiple_of(1000) {
        eprintln!("[strfree] count={n}");
    }
}

#[no_mangle]
pub extern "C" fn haxe_string_free(s: *mut HaxeString) {
    if s.is_null() {
        return;
    }
    strfree_dbg_count();

    unsafe {
        // Calling this twice on one handle is a DOUBLE FREE, and nothing here
        // can make it safe: the header itself is reclaimed below, so a second
        // call reads and writes memory that is already back with the allocator.
        //
        // Clearing the fields first is damage limitation, not a guarantee. It
        // holds only while the freed header happens to be untouched, and buys
        // exactly one thing: a second call sees a null buffer pointer and so
        // does not release the byte buffer twice as well. The caller is still
        // responsible for releasing a string once.
        // Under `RZT_DBG_STRFREE` the header is KEPT rather than reclaimed, so
        // the mark below survives and a second release on the same handle can
        // be named instead of corrupting the heap silently. That leaks a header
        // per string, which is the price of the diagnostic; it is off by
        // default and must stay off outside an investigation.
        if strfree_keep_headers() {
            if (*s).len == FREED_MARK {
                eprintln!("[strfree] DOUBLE FREE of {s:p} — already released");
                return;
            }
            let buf_ptr = (*s).ptr;
            let buf_cap = (*s).cap;
            (*s).ptr = std::ptr::null_mut();
            (*s).len = FREED_MARK;
            (*s).cap = 0;
            if !buf_ptr.is_null() && buf_cap > 0 {
                dealloc(buf_ptr, Layout::from_size_align_unchecked(buf_cap, 1));
            }
            return;
        }

        let buf_ptr = (*s).ptr;
        let buf_cap = (*s).cap;
        (*s).ptr = std::ptr::null_mut();
        (*s).len = 0;
        (*s).cap = 0;

        if !buf_ptr.is_null() && buf_cap > 0 {
            let layout = Layout::from_size_align_unchecked(buf_cap, 1);
            dealloc(buf_ptr, layout);
        }

        // Reclaim the heap-allocated HaxeString header. Safe because every
        // exported producer routes through `Box::into_raw(Box::new(..))`.
        drop(Box::from_raw(s));
    }
}

// ============================================================================
// I/O and Conversion
// ============================================================================

/// Print string to stdout
#[no_mangle]
/// `Sys.print`/`Sys.println` are declared `(v:Dynamic)`, so the compiler boxes
/// a non-String argument (`haxe_box_int_ptr` etc.) but passes a String RAW as a
/// `*HaxeString`. Both then arrive here. Reading a boxed `DynamicValue` as a
/// `HaxeString` dereferences its type tag as a data pointer, which is why
/// `Sys.println(42)` segfaulted while `Sys.println("x=" + 42)` worked.
///
/// Discriminating is safe because the two layouts differ in their first eight
/// bytes: `DynamicValue` starts with a small `type_id` (0..=7 for the builtin
/// scalars), while `HaxeString` starts with a heap pointer whose low 32 bits
/// are never that small. A tag outside that window is left alone and treated as
/// a string, so boxed user objects behave exactly as before.
///
/// Returns an owned HaxeString when `s` was a boxed scalar, else None.
unsafe fn dynamic_box_to_string(s: *const HaxeString) -> Option<*mut HaxeString> {
    use crate::type_system::{DynamicValue, TYPE_ARRAY};
    if (s as usize) < 0x1000 || (s as usize) & 7 != 0 {
        return None;
    }
    let d = *(s as *const DynamicValue);
    if d.type_id.0 > TYPE_ARRAY.0 {
        return None;
    }
    Some(crate::type_system::haxe_std_string_ptr(s as *mut u8))
}

/// The printable bytes of a string, unwrapping a boxed Dynamic and rejecting
/// invalid UTF-8, so print and println agree on what they are writing.
///
/// # Safety
/// `s` must be null or a valid `HaxeString` whose `ptr`/`len` describe live
/// memory for the duration of the borrow.
unsafe fn printable_bytes<'a>(s: *const HaxeString) -> Option<&'a [u8]> {
    if s.is_null() {
        return None;
    }
    if let Some(boxed) = dynamic_box_to_string(s) {
        return printable_bytes(boxed);
    }
    let s_ref = &*s;
    if s_ref.len == 0 {
        return None;
    }
    let slice = slice::from_raw_parts(s_ref.ptr, s_ref.len);
    if str::from_utf8(slice).is_ok() {
        Some(slice)
    } else {
        None
    }
}

pub extern "C" fn haxe_string_print(s: *const HaxeString) {
    if let Some(slice) = unsafe { printable_bytes(s) } {
        write_stdout_buffered(slice, slice.contains(&b'\n'));
    }
}

/// Print string to stdout with newline
#[no_mangle]
pub extern "C" fn haxe_string_println(s: *const HaxeString) {
    // One write, so the newline cannot be separated from its line. An empty or
    // unprintable argument still emits the newline, as it did when this was two
    // calls and the first one returned early.
    let slice = unsafe { printable_bytes(s) }.unwrap_or(b"");
    write_stdout_parts(slice, b"\n", true);
}

/// Replace all occurrences of `needle` in `haystack` with `replacement`.
/// Returns a new HaxeString with the result.
#[no_mangle]
pub extern "C" fn haxe_string_replace(
    haystack: *const HaxeString,
    needle: *const HaxeString,
    replacement: *const HaxeString,
) -> *mut HaxeString {
    let result = Box::new(HaxeString {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    });
    let result_ptr = Box::into_raw(result);
    haxe_string_new(result_ptr);

    if haystack.is_null() || needle.is_null() || replacement.is_null() {
        return result_ptr;
    }

    unsafe {
        let h = &*haystack;
        let n = &*needle;
        let r = &*replacement;

        if h.len == 0 || n.len == 0 || h.ptr.is_null() || n.ptr.is_null() {
            // Copy haystack as-is
            if h.len > 0 && !h.ptr.is_null() {
                let h_slice = slice::from_raw_parts(h.ptr, h.len);
                haxe_string_from_bytes(result_ptr, h_slice.as_ptr(), h_slice.len());
            }
            return result_ptr;
        }

        let h_bytes = slice::from_raw_parts(h.ptr, h.len);
        let n_bytes = slice::from_raw_parts(n.ptr, n.len);
        let r_bytes = if r.len > 0 && !r.ptr.is_null() {
            slice::from_raw_parts(r.ptr, r.len)
        } else {
            &[]
        };

        // Simple search-and-replace
        let h_str = str::from_utf8_unchecked(h_bytes);
        let n_str = str::from_utf8_unchecked(n_bytes);
        let r_str = str::from_utf8_unchecked(r_bytes);
        let replaced = h_str.replace(n_str, r_str);

        haxe_string_from_bytes(result_ptr, replaced.as_ptr(), replaced.len());
        result_ptr
    }
}

/// Get C string pointer (null-terminated)
#[no_mangle]
pub extern "C" fn haxe_string_to_cstr(s: *const HaxeString) -> *const u8 {
    if s.is_null() {
        return ptr::null();
    }
    unsafe { (*s).ptr }
}

/// Hash a string using FNV-1a. Returns i32 for Haxe Int compatibility.
#[no_mangle]
pub extern "C" fn haxe_string_hash(s: *const HaxeString) -> i32 {
    if s.is_null() {
        return 0;
    }
    unsafe {
        let hs = &*s;
        if hs.ptr.is_null() || hs.len == 0 {
            return 0;
        }
        let bytes = slice::from_raw_parts(hs.ptr, hs.len);
        // FNV-1a hash
        let mut hash: u32 = 2166136261;
        for &b in bytes {
            hash ^= b as u32;
            hash = hash.wrapping_mul(16777619);
        }
        hash as i32
    }
}
