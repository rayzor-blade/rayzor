//! DEFLATE compression/decompression runtime for haxe.zip.Compress / Uncompress
//!
//! Backed by the `flate2` crate. Provides both streaming (execute) and
//! one-shot (run) APIs matching the Haxe stdlib interface.

use crate::anon_object::{rayzor_anon_new, rayzor_anon_set_field_by_index, rayzor_ensure_shape};
use crate::haxe_sys::{haxe_bytes_alloc, HaxeBytes};
use std::sync::Once;

// Shape ID for {done:Bool, read:Int, write:Int} anonymous return.
// Fields sorted alphabetically: done (index 0), read (index 1), write (index 2).
// Type codes: 1 = Bool, 3 = Int
const EXECUTE_RESULT_SHAPE_ID: u32 = 1001;

static SHAPE_INIT: Once = Once::new();

fn ensure_execute_result_shape() {
    SHAPE_INIT.call_once(|| {
        let descriptor = crate::ereg::rust_str_to_hs("done:1,read:3,write:3");
        rayzor_ensure_shape(EXECUTE_RESULT_SHAPE_ID, descriptor);
    });
}

/// Build the anonymous result object {done:Bool, read:Int, write:Int}
fn build_execute_result(done: bool, read: i64, write: i64) -> *mut u8 {
    ensure_execute_result_shape();
    let handle = rayzor_anon_new(EXECUTE_RESULT_SHAPE_ID, 3);
    rayzor_anon_set_field_by_index(handle, 0, if done { 1 } else { 0 }); // done
    rayzor_anon_set_field_by_index(handle, 1, read as u64); // read
    rayzor_anon_set_field_by_index(handle, 2, write as u64); // write
    handle
}

// ============================================================================
// Compress (DEFLATE encoding)
// ============================================================================

struct CompressHandle {
    inner: flate2::Compress,
    flush: flate2::FlushCompress,
}

/// Map Haxe FlushMode enum ordinal to flate2 FlushCompress
fn to_flush_compress(mode: i32) -> flate2::FlushCompress {
    match mode {
        0 => flate2::FlushCompress::None,
        1 => flate2::FlushCompress::Sync,
        2 => flate2::FlushCompress::Full,
        3 => flate2::FlushCompress::Finish,
        // BLOCK not directly supported — map to Sync
        4 => flate2::FlushCompress::Sync,
        _ => flate2::FlushCompress::None,
    }
}

/// Compress.new(level:Int) -> handle
#[no_mangle]
pub extern "C" fn rayzor_compress_new(level: i32) -> *mut u8 {
    let level = flate2::Compression::new(level.clamp(0, 9) as u32);
    let handle = Box::new(CompressHandle {
        inner: flate2::Compress::new(level, true), // zlib wrapper = true
        flush: flate2::FlushCompress::None,
    });
    Box::into_raw(handle) as *mut u8
}

/// Compress.execute(src, srcPos, dst, dstPos) -> {done, read, write}
#[no_mangle]
pub extern "C" fn rayzor_compress_execute(
    handle: *mut u8,
    src_bytes: *mut u8,
    src_pos: i32,
    dst_bytes: *mut u8,
    dst_pos: i32,
) -> *mut u8 {
    if handle.is_null() || src_bytes.is_null() || dst_bytes.is_null() {
        return build_execute_result(false, 0, 0);
    }

    unsafe {
        let compress = &mut *(handle as *mut CompressHandle);
        let src = &*(src_bytes as *const HaxeBytes);
        let dst = &mut *(dst_bytes as *mut HaxeBytes);

        let src_pos = src_pos.max(0) as usize;
        let dst_pos = dst_pos.max(0) as usize;

        if src_pos >= src.len || dst_pos >= dst.len {
            return build_execute_result(false, 0, 0);
        }

        let input = std::slice::from_raw_parts(src.ptr.add(src_pos), src.len - src_pos);
        let output = std::slice::from_raw_parts_mut(dst.ptr.add(dst_pos), dst.len - dst_pos);

        let before_in = compress.inner.total_in();
        let before_out = compress.inner.total_out();

        let status = compress.inner.compress(input, output, compress.flush);

        let bytes_read = (compress.inner.total_in() - before_in) as i64;
        let bytes_written = (compress.inner.total_out() - before_out) as i64;

        let done = matches!(status, Ok(flate2::Status::StreamEnd));

        build_execute_result(done, bytes_read, bytes_written)
    }
}

/// Compress.setFlushMode(mode)
#[no_mangle]
pub extern "C" fn rayzor_compress_set_flush(handle: *mut u8, mode: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let compress = &mut *(handle as *mut CompressHandle);
        compress.flush = to_flush_compress(mode);
    }
}

/// Compress.close()
#[no_mangle]
pub extern "C" fn rayzor_compress_close(handle: *mut u8) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut CompressHandle));
    }
}

/// Compress.run(bytes, level) -> compressed Bytes
/// One-shot convenience: compress entire input buffer, return new Bytes.
#[no_mangle]
pub extern "C" fn rayzor_compress_run(src_bytes: *mut u8, level: i32) -> *mut u8 {
    if src_bytes.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let src = &*(src_bytes as *const HaxeBytes);
        let input = std::slice::from_raw_parts(src.ptr, src.len);

        let level = flate2::Compression::new(level.clamp(0, 9) as u32);
        let mut encoder = flate2::Compress::new(level, true); // zlib wrapper

        // Worst case: zlib output can be slightly larger than input + header/trailer
        let max_out = input.len() + input.len() / 100 + 64;
        let mut output = vec![0u8; max_out];

        let status = encoder.compress(input, &mut output, flate2::FlushCompress::Finish);
        match status {
            Ok(flate2::Status::StreamEnd) => {}
            _ => {
                // Retry with larger buffer
                output.resize(max_out * 2, 0);
                let _ = encoder.compress(input, &mut output, flate2::FlushCompress::Finish);
            }
        }

        let written = encoder.total_out() as usize;

        // Allocate HaxeBytes and copy compressed data
        let result = haxe_bytes_alloc(written as i32);
        if result.is_null() {
            return std::ptr::null_mut();
        }
        std::ptr::copy_nonoverlapping(output.as_ptr(), (*result).ptr, written);

        result as *mut u8
    }
}

// ============================================================================
// Uncompress (DEFLATE decoding)
// ============================================================================

struct UncompressHandle {
    inner: flate2::Decompress,
    flush: flate2::FlushDecompress,
}

/// Map Haxe FlushMode enum ordinal to flate2 FlushDecompress
fn to_flush_decompress(mode: i32) -> flate2::FlushDecompress {
    match mode {
        0 => flate2::FlushDecompress::None,
        1 => flate2::FlushDecompress::Sync,
        2 => flate2::FlushDecompress::Finish,
        3 => flate2::FlushDecompress::Finish,
        _ => flate2::FlushDecompress::None,
    }
}

/// Uncompress.new(?windowBits) -> handle
#[no_mangle]
pub extern "C" fn rayzor_uncompress_new(window_bits: i32) -> *mut u8 {
    // window_bits: 0 or negative means raw deflate, positive means zlib wrapper
    let zlib = window_bits >= 0;
    let handle = Box::new(UncompressHandle {
        inner: flate2::Decompress::new(zlib),
        flush: flate2::FlushDecompress::None,
    });
    Box::into_raw(handle) as *mut u8
}

/// Uncompress.execute(src, srcPos, dst, dstPos) -> {done, read, write}
#[no_mangle]
pub extern "C" fn rayzor_uncompress_execute(
    handle: *mut u8,
    src_bytes: *mut u8,
    src_pos: i32,
    dst_bytes: *mut u8,
    dst_pos: i32,
) -> *mut u8 {
    if handle.is_null() || src_bytes.is_null() || dst_bytes.is_null() {
        return build_execute_result(false, 0, 0);
    }

    unsafe {
        let decompress = &mut *(handle as *mut UncompressHandle);
        let src = &*(src_bytes as *const HaxeBytes);
        let dst = &mut *(dst_bytes as *mut HaxeBytes);

        let src_pos = src_pos.max(0) as usize;
        let dst_pos = dst_pos.max(0) as usize;

        if src_pos >= src.len || dst_pos >= dst.len {
            return build_execute_result(false, 0, 0);
        }

        let input = std::slice::from_raw_parts(src.ptr.add(src_pos), src.len - src_pos);
        let output = std::slice::from_raw_parts_mut(dst.ptr.add(dst_pos), dst.len - dst_pos);

        let before_in = decompress.inner.total_in();
        let before_out = decompress.inner.total_out();

        let status = decompress.inner.decompress(input, output, decompress.flush);

        let bytes_read = (decompress.inner.total_in() - before_in) as i64;
        let bytes_written = (decompress.inner.total_out() - before_out) as i64;

        let done = matches!(status, Ok(flate2::Status::StreamEnd));

        build_execute_result(done, bytes_read, bytes_written)
    }
}

/// Uncompress.setFlushMode(mode)
#[no_mangle]
pub extern "C" fn rayzor_uncompress_set_flush(handle: *mut u8, mode: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let decompress = &mut *(handle as *mut UncompressHandle);
        decompress.flush = to_flush_decompress(mode);
    }
}

/// Uncompress.close()
#[no_mangle]
pub extern "C" fn rayzor_uncompress_close(handle: *mut u8) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut UncompressHandle));
    }
}
