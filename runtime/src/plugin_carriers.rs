//! Stable host access to Rayzor strings and bytes for native packages.

use crate::haxe_string::HaxeString;
use crate::haxe_sys::HaxeBytes;
use std::ptr;

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_string_data(value: *const u8) -> *const u8 {
    if value.is_null() {
        return ptr::null();
    }
    unsafe { (*(value as *const HaxeString)).ptr }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_string_len(value: *const u8) -> usize {
    if value.is_null() {
        return 0;
    }
    unsafe { (*(value as *const HaxeString)).len }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_string_new(data: *const u8, len: usize) -> *mut u8 {
    let out = Box::into_raw(Box::new(HaxeString {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    }));
    crate::haxe_string::haxe_string_from_bytes(out, data, len);
    out.cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_bytes_data(value: *const u8) -> *const u8 {
    if value.is_null() {
        return ptr::null();
    }
    unsafe { (*(value as *const HaxeBytes)).ptr }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_bytes_data_mut(value: *mut u8) -> *mut u8 {
    if value.is_null() {
        return ptr::null_mut();
    }
    unsafe { (*(value as *mut HaxeBytes)).ptr }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_bytes_len(value: *const u8) -> usize {
    if value.is_null() {
        return 0;
    }
    unsafe { (*(value as *const HaxeBytes)).len }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_bytes_new(data: *const u8, len: usize) -> *mut u8 {
    let Ok(size) = i32::try_from(len) else {
        return ptr::null_mut();
    };
    let out = crate::haxe_sys::haxe_bytes_alloc(size);
    if out.is_null() || data.is_null() || len == 0 {
        return out.cast();
    }
    unsafe { ptr::copy_nonoverlapping(data, (*out).ptr, len) };
    out.cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_bytes_retain(value: *mut u8) -> *mut u8 {
    if value.is_null() {
        return ptr::null_mut();
    }
    let len = rayzor_plugin_bytes_len(value);
    let Ok(len) = i64::try_from(len) else {
        return ptr::null_mut();
    };
    crate::haxe_sys::haxe_bytes_sub_i64(value.cast(), 0, len).cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_bytes_release(value: *mut u8) {
    crate::haxe_sys::haxe_bytes_free(value.cast());
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_raise(message: *const u8, message_len: usize) -> ! {
    let message = if message.is_null() || message_len == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(message, message_len) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    crate::exception::throw_with_message(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_and_bytes_round_trip_without_an_intermediate_encoding() {
        let source = b"gpu adapter";
        let text = rayzor_plugin_string_new(source.as_ptr(), source.len());
        assert_eq!(rayzor_plugin_string_len(text), source.len());
        assert_eq!(
            unsafe { std::slice::from_raw_parts(rayzor_plugin_string_data(text), source.len()) },
            source
        );
        crate::haxe_string::haxe_string_free(text.cast());

        let bytes = rayzor_plugin_bytes_new(source.as_ptr(), source.len());
        assert_eq!(rayzor_plugin_bytes_len(bytes), source.len());
        assert_eq!(
            unsafe { std::slice::from_raw_parts(rayzor_plugin_bytes_data(bytes), source.len()) },
            source
        );
        let retained = rayzor_plugin_bytes_retain(bytes);
        rayzor_plugin_bytes_release(bytes);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(rayzor_plugin_bytes_data(retained), source.len()) },
            source
        );
        rayzor_plugin_bytes_release(retained);
    }
}
