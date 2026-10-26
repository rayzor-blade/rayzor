//! Haxe `Bytes`, `FileInput`/`FileOutput` streams and `Sys` std streams.
//!
//! Bytes header (40 bytes, the wasm32 layout of the native `HaxeBytes` and the
//! one `wasm_runner::write_haxe_bytes_header` writes):
//! `{ptr: u32, len: u32, cap: u32, kind: u8, pad[7], owner: u32, refcount: i64,
//! fd: i32, pad[4]}`. Data comes from `rt_alloc` (align 8) with `cap` its exact
//! size, so `haxe_bytes_free` can dealloc it.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};

use crate::{
    haxe_array_get_i64, haxe_array_length, haxe_array_new, haxe_array_push_i64, rayzor_throw_typed,
    read_haxe_string, rt_alloc, wasi_write, wasm_string_of,
};

const BYTES_HEADER: usize = 40;
const BYTES_KIND_MALLOC: u8 = 0;
const BYTES_KIND_VIEW: u8 = 2;
/// Type id `rayzor_throw_typed` prints as a String message.
const STRING_TYPE_ID: u32 = 5;

const SEEK_BEGIN: i32 = 0;
const SEEK_CUR: i32 = 1;
const SEEK_END: i32 = 2;

fn throw_message(msg: &[u8]) -> ! {
    rayzor_throw_typed(wasm_string_of(msg), STRING_TYPE_ID);
    unreachable!()
}

fn write_bytes_header(header: i32, data: u32, len: u32, cap: u32, kind: u8, owner: u32) {
    unsafe {
        let h = header as *mut u8;
        std::ptr::write_bytes(h, 0, BYTES_HEADER);
        *(h as *mut u32) = data;
        *(h.add(4) as *mut u32) = len;
        *(h.add(8) as *mut u32) = cap;
        *h.add(12) = kind;
        *(h.add(20) as *mut u32) = owner;
        *(h.add(24) as *mut i64) = 1;
        *(h.add(32) as *mut i32) = -1;
    }
}

/// A zero-filled owner Bytes of `len` bytes; 0 on allocation failure.
fn new_bytes(len: usize) -> i32 {
    // Never a zero-size block, so `ptr` is always valid and `cap` always matches it.
    let cap = len.max(1);
    let data = rt_alloc(cap);
    let header = rt_alloc(BYTES_HEADER);
    if data == 0 || header == 0 {
        return 0;
    }
    unsafe { std::ptr::write_bytes(data as *mut u8, 0, cap) };
    write_bytes_header(
        header,
        data as u32,
        len as u32,
        cap as u32,
        BYTES_KIND_MALLOC,
        0,
    );
    header
}

fn bytes_from_slice(src: &[u8]) -> i32 {
    let b = new_bytes(src.len());
    if b != 0 && !src.is_empty() {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), bytes_parts(b).0, src.len());
        }
    }
    b
}

/// `(data, len)` of a non-null Bytes header.
fn bytes_parts(b: i32) -> (*mut u8, usize) {
    unsafe {
        let h = b as *const u32;
        (*h as *mut u8, *h.add(1) as usize)
    }
}

fn bytes_slice<'a>(b: i32) -> &'a mut [u8] {
    let (data, len) = bytes_parts(b);
    if data.is_null() || len == 0 {
        return &mut [];
    }
    unsafe { std::slice::from_raw_parts_mut(data, len) }
}

fn string_bytes<'a>(s: i32) -> Option<&'a [u8]> {
    if s == 0 {
        return None;
    }
    let (data, len, _) = unsafe { read_haxe_string(s) };
    if data == 0 {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(data as *const u8, len as usize) })
}

/// `pos..pos+len` within a Bytes of length `total`, if in range.
fn range(pos: i32, len: i32, total: usize) -> Option<std::ops::Range<usize>> {
    if pos < 0 || len < 0 {
        return None;
    }
    let (start, count) = (pos as usize, len as usize);
    (start <= total && count <= total - start).then(|| start..start + count)
}

// ============================================================================
// Bytes
// ============================================================================

/// `Bytes.alloc(size)`: zero-filled.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_alloc(size: i32) -> i32 {
    new_bytes(size.max(0) as usize)
}

/// `new Bytes(length, data)` over an `Array<Int>` of byte values.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_of_int_array(length: i32, data: i32) -> i32 {
    let b = new_bytes(length.max(0) as usize);
    if b == 0 || data == 0 {
        return b;
    }
    let n = (haxe_array_length(data).max(0) as usize).min(length.max(0) as usize);
    let dst = bytes_slice(b);
    for (i, slot) in dst.iter_mut().enumerate().take(n) {
        *slot = haxe_array_get_i64(data, i as i32) as u8;
    }
    b
}

/// `Bytes.ofString(s)`: the UTF-8 bytes of `s`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_of_string(s: i32) -> i32 {
    bytes_from_slice(string_bytes(s).unwrap_or(&[]))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_length(b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    bytes_parts(b).1 as i32
}

/// `bytes.getData()`: an `Array<Int>` copy; later writes to either side do not
/// propagate.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_get_data(b: i32) -> i32 {
    let arr = haxe_array_new();
    if b == 0 || arr == 0 {
        return arr;
    }
    for &v in bytes_slice(b).iter() {
        haxe_array_push_i64(arr, v as i32);
    }
    arr
}

/// `bytes.get(pos)`: 0 when out of range.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_get(b: i32, pos: i32) -> i32 {
    if b == 0 || pos < 0 {
        return 0;
    }
    bytes_slice(b).get(pos as usize).map_or(0, |&v| v as i32)
}

/// `bytes.set(pos, v)`: ignored when out of range.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_set(b: i32, pos: i32, value: i32) {
    if b == 0 || pos < 0 {
        return;
    }
    if let Some(slot) = bytes_slice(b).get_mut(pos as usize) {
        *slot = value as u8;
    }
}

/// `bytes.sub(pos, len)`: a view sharing the owner's data, as the native
/// runtime and the wasm host do. Never pass a view to `haxe_bytes_free`.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_sub(b: i32, pos: i32, len: i32) -> i32 {
    if b == 0 {
        throw_message(b"Outside bounds");
    }
    let (data, total) = bytes_parts(b);
    let Some(r) = range(pos, len, total) else {
        throw_message(b"Outside bounds");
    };
    let header = rt_alloc(BYTES_HEADER);
    if header == 0 {
        return 0;
    }
    let view = unsafe { data.add(r.start) } as u32;
    write_bytes_header(
        header,
        view,
        r.len() as u32,
        r.len() as u32,
        BYTES_KIND_VIEW,
        b as u32,
    );
    header
}

/// `src.blitTo(srcPos, dest, destPos, len)`: memmove; ignored when out of range.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_blit(src: i32, src_pos: i32, dest: i32, dest_pos: i32, len: i32) {
    if src == 0 || dest == 0 || len <= 0 {
        return;
    }
    let (s, s_len) = bytes_parts(src);
    let (d, d_len) = bytes_parts(dest);
    let (Some(sr), Some(dr)) = (range(src_pos, len, s_len), range(dest_pos, len, d_len)) else {
        return;
    };
    unsafe { std::ptr::copy(s.add(sr.start), d.add(dr.start), len as usize) };
}

/// `bytes.toString()`: lossy UTF-8.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_to_string(b: i32) -> i32 {
    if b == 0 {
        return wasm_string_of(&[]);
    }
    wasm_string_of(String::from_utf8_lossy(bytes_slice(b)).as_bytes())
}

/// `bytes.getString(pos, len)`: lossy UTF-8; throws when out of range.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_bytes_get_string(b: i32, pos: i32, len: i32) -> i32 {
    if b == 0 {
        throw_message(b"Outside bounds");
    }
    let data = bytes_slice(b);
    let Some(r) = range(pos, len, data.len()) else {
        throw_message(b"Outside bounds");
    };
    wasm_string_of(String::from_utf8_lossy(&data[r]).as_bytes())
}

// ============================================================================
// Streams
// ============================================================================
//
// A handle is a boxed `InStream` / `OutStream` whose first field is a tag, so a
// pointer that is not one of these (or is null) degrades to a no-op instead of
// being reinterpreted.

const IN_TAG: u32 = 0x5259_4649; // "RYFI"
const OUT_TAG: u32 = 0x5259_464F; // "RYFO"

enum InSource {
    Stdin(BufReader<std::io::Stdin>),
    File(BufReader<File>),
}

#[repr(C)]
struct InStream {
    tag: u32,
    eof: bool,
    src: InSource,
}

impl InSource {
    fn reader(&mut self) -> &mut dyn BufRead {
        match self {
            InSource::Stdin(r) => r,
            InSource::File(r) => r,
        }
    }
}

enum OutSink {
    /// Unbuffered raw fd, so output interleaves with the runtime's own `fd_write`s.
    Fd(i32),
    File(BufWriter<File>),
}

#[repr(C)]
struct OutStream {
    tag: u32,
    sink: OutSink,
}

impl OutSink {
    fn write(&mut self, data: &[u8]) -> usize {
        match self {
            OutSink::Fd(fd) => {
                unsafe { wasi_write(*fd, data.as_ptr(), data.len()) };
                data.len()
            }
            OutSink::File(w) => w.write(data).unwrap_or(0),
        }
    }
}

fn in_stream<'a>(h: i32) -> Option<&'a mut InStream> {
    if h <= 0 {
        return None;
    }
    let s = h as *mut InStream;
    unsafe { ((*s).tag == IN_TAG).then(|| &mut *s) }
}

fn out_stream<'a>(h: i32) -> Option<&'a mut OutStream> {
    if h <= 0 {
        return None;
    }
    let s = h as *mut OutStream;
    unsafe { ((*s).tag == OUT_TAG).then(|| &mut *s) }
}

fn seek_from(p: i32, whence: i32) -> Option<SeekFrom> {
    match whence {
        SEEK_BEGIN => Some(SeekFrom::Start(p.max(0) as u64)),
        SEEK_CUR => Some(SeekFrom::Current(p as i64)),
        SEEK_END => Some(SeekFrom::End(p as i64)),
        _ => None,
    }
}

fn throw_eof(s: &mut InStream) -> ! {
    s.eof = true;
    throw_message(b"Eof")
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileinput_read_byte(h: i32) -> i32 {
    let Some(s) = in_stream(h) else {
        return -1;
    };
    let mut buf = [0u8; 1];
    match s.src.reader().read(&mut buf) {
        Ok(1) => buf[0] as i32,
        _ => throw_eof(s),
    }
}

/// `input.readBytes(bytes, pos, len)`: one read, returns the count; throws Eof
/// when nothing could be read.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileinput_read_bytes_buf(h: i32, bytes: i32, pos: i32, len: i32) -> i32 {
    let Some(s) = in_stream(h) else {
        return 0;
    };
    if bytes == 0 || len <= 0 {
        return 0;
    }
    let data = bytes_slice(bytes);
    let Some(r) = range(pos, len, data.len()) else {
        return 0;
    };
    match s.src.reader().read(&mut data[r]) {
        Ok(n) if n > 0 => n as i32,
        _ => throw_eof(s),
    }
}

/// `input.readLine()`: strips `\n` / `\r\n`; at end of input returns "" and sets eof.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileinput_read_line(h: i32) -> i32 {
    let Some(s) = in_stream(h) else {
        return wasm_string_of(&[]);
    };
    let mut line = Vec::new();
    match s.src.reader().read_until(b'\n', &mut line) {
        Ok(n) if n > 0 => {
            if line.last() == Some(&b'\n') {
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
            }
            wasm_string_of(String::from_utf8_lossy(&line).as_bytes())
        }
        _ => {
            s.eof = true;
            wasm_string_of(&[])
        }
    }
}

/// `input.readAll()`: the rest of the input as Bytes.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileinput_read_all(h: i32) -> i32 {
    let Some(s) = in_stream(h) else {
        return new_bytes(0);
    };
    let mut buf = Vec::new();
    let _ = s.src.reader().read_to_end(&mut buf);
    s.eof = true;
    bytes_from_slice(&buf)
}

/// `input.seek(p, whence)`: a no-op on stdin; clears eof.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileinput_seek(h: i32, p: i32, whence: i32) {
    let Some(s) = in_stream(h) else {
        return;
    };
    let Some(to) = seek_from(p, whence) else {
        return;
    };
    if let InSource::File(r) = &mut s.src {
        let _ = r.seek(to);
    }
    s.eof = false;
}

/// `input.tell()`: 0 on stdin.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileinput_tell(h: i32) -> i32 {
    match in_stream(h).map(|s| &mut s.src) {
        Some(InSource::File(r)) => r.stream_position().map_or(0, |p| p as i32),
        _ => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileinput_eof(h: i32) -> i32 {
    in_stream(h).is_none_or(|s| s.eof) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileinput_close(h: i32) {
    if let Some(s) = in_stream(h) {
        s.tag = 0;
        drop(unsafe { Box::from_raw(s as *mut InStream) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileoutput_write_byte(h: i32, c: i32) {
    if let Some(s) = out_stream(h) {
        s.sink.write(&[c as u8]);
    }
}

/// `output.writeBytes(bytes, pos, len)`: returns the count written, 0 when out of range.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileoutput_write_bytes_buf(h: i32, bytes: i32, pos: i32, len: i32) -> i32 {
    let Some(s) = out_stream(h) else {
        return 0;
    };
    if bytes == 0 || len <= 0 {
        return 0;
    }
    let data = bytes_slice(bytes);
    match range(pos, len, data.len()) {
        Some(r) => s.sink.write(&data[r]) as i32,
        None => 0,
    }
}

/// `output.seek(p, whence)`: flushes first; a no-op on stdout/stderr.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileoutput_seek(h: i32, p: i32, whence: i32) {
    if let (Some(OutSink::File(w)), Some(to)) =
        (out_stream(h).map(|s| &mut s.sink), seek_from(p, whence))
    {
        let _ = w.flush();
        let _ = w.seek(to);
    }
}

/// `output.tell()`: 0 on stdout/stderr.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileoutput_tell(h: i32) -> i32 {
    match out_stream(h).map(|s| &mut s.sink) {
        Some(OutSink::File(w)) => w.stream_position().map_or(0, |p| p as i32),
        _ => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileoutput_flush(h: i32) {
    if let Some(OutSink::File(w)) = out_stream(h).map(|s| &mut s.sink) {
        let _ = w.flush();
    }
}

/// Flushes and frees the handle; closing a std stream leaves the process fd open.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_fileoutput_close(h: i32) {
    if let Some(s) = out_stream(h) {
        s.tag = 0;
        let mut owned = unsafe { Box::from_raw(s as *mut OutStream) };
        if let OutSink::File(w) = &mut owned.sink {
            let _ = w.flush();
        }
    }
}

fn box_input(src: InSource) -> i32 {
    Box::into_raw(Box::new(InStream {
        tag: IN_TAG,
        eof: false,
        src,
    })) as i32
}

fn box_output(sink: OutSink) -> i32 {
    Box::into_raw(Box::new(OutStream { tag: OUT_TAG, sink })) as i32
}

fn path_of<'a>(path: i32) -> Option<&'a str> {
    string_bytes(path).and_then(|p| std::str::from_utf8(p).ok())
}

/// A FileInput handle over `path`, 0 when it cannot be opened (`File.read`).
pub(crate) fn open_input(path: i32) -> i32 {
    match path_of(path).map(File::open) {
        Some(Ok(f)) => box_input(InSource::File(BufReader::new(f))),
        _ => 0,
    }
}

/// A FileOutput handle over `path` opened with `opts`, 0 on failure
/// (`File.write` / `append` / `update`).
pub(crate) fn open_output(path: i32, opts: &std::fs::OpenOptions) -> i32 {
    match path_of(path).map(|p| opts.open(p)) {
        Some(Ok(f)) => box_output(OutSink::File(BufWriter::new(f))),
        _ => 0,
    }
}

// ============================================================================
// Sys
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn haxe_sys_stdin() -> i32 {
    box_input(InSource::Stdin(BufReader::new(std::io::stdin())))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_sys_stdout() -> i32 {
    box_output(OutSink::Fd(1))
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_sys_stderr() -> i32 {
    box_output(OutSink::Fd(2))
}

/// `Sys.getEnv(name)`: null when unset, empty, or not UTF-8.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_sys_get_env(name: i32) -> i32 {
    let Some(name) = string_bytes(name).filter(|n| !n.is_empty()) else {
        return 0;
    };
    let Ok(name) = std::str::from_utf8(name) else {
        return 0;
    };
    match std::env::var(name) {
        Ok(v) => wasm_string_of(v.as_bytes()),
        Err(_) => 0,
    }
}

/// `Sys.time()`: seconds since the Unix epoch.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_sys_time() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// `Sys.cpuTime()`: seconds since the first call, as the native runtime.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_sys_cpu_time() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}
