//! rayzor-window — cross-platform native windowing via raw system FFI.
//!
//! Zero third-party dependencies. Uses system APIs directly:
//! - macOS: objc_msgSend + Cocoa (libobjc.dylib)
//! - Linux: dlopen("libX11.so") + X11
//! - Windows: CreateWindowExW (user32.dll)

use std::ffi::c_void;

#[cfg(target_os = "macos")]
mod cocoa;

use rayzor_plugin::NativeMethodDesc;
use rayzor_runtime::haxe_string::HaxeString;

// ============================================================================
// NativeWindow — platform-agnostic wrapper
// ============================================================================

pub struct NativeWindow {
    #[cfg(target_os = "macos")]
    inner: cocoa::CocoaWindow,
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_create(
    title: *const HaxeString,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    style: i32,
) -> *mut NativeWindow {
    let title_str = haxe_string_to_str(title);

    #[cfg(target_os = "macos")]
    {
        match cocoa::CocoaWindow::create(&title_str, x, y, w, h, style) {
            Some(win) => Box::into_raw(Box::new(NativeWindow { inner: win })),
            None => std::ptr::null_mut(),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("[rayzor-window] Platform not yet supported");
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_create_centered(
    title: *const HaxeString,
    w: i32,
    h: i32,
) -> *mut NativeWindow {
    let title_str = haxe_string_to_str(title);

    #[cfg(target_os = "macos")]
    {
        match cocoa::CocoaWindow::create_centered(&title_str, w, h) {
            Some(win) => Box::into_raw(Box::new(NativeWindow { inner: win })),
            None => std::ptr::null_mut(),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("[rayzor-window] Platform not yet supported");
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_poll_events(win: *mut NativeWindow) -> i32 {
    if win.is_null() { return 0; }
    let win = &mut *win;
    if win.inner.poll_events() { 1 } else { 0 }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_is_key_down(win: *mut NativeWindow, key: i32) -> i32 {
    if win.is_null() { return 0; }
    if (*win).inner.is_key_down(key) { 1 } else { 0 }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_handle(win: *mut NativeWindow) -> *mut c_void {
    if win.is_null() { return std::ptr::null_mut(); }
    (*win).inner.ns_view
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_display_handle(win: *mut NativeWindow) -> *mut c_void {
    // macOS/Windows: null (AppKit/Win32 don't need separate display handle)
    // Linux: would return X11 Display*
    std::ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_width(win: *mut NativeWindow) -> i32 {
    if win.is_null() { return 0; }
    (*win).inner.width as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_height(win: *mut NativeWindow) -> i32 {
    if win.is_null() { return 0; }
    (*win).inner.height as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_was_resized(win: *mut NativeWindow) -> i32 {
    if win.is_null() { return 0; }
    if (*win).inner.resized { 1 } else { 0 }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_title(
    win: *mut NativeWindow,
    title: *const HaxeString,
) {
    if win.is_null() { return; }
    let title_str = haxe_string_to_str(title);
    (*win).inner.set_title(&title_str);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_destroy(win: *mut NativeWindow) {
    if !win.is_null() {
        (*win).inner.destroy();
        drop(Box::from_raw(win));
    }
}

// ============================================================================
// HaxeString helper
// ============================================================================

unsafe fn haxe_string_to_str(s: *const HaxeString) -> String {
    if s.is_null() {
        return String::new();
    }
    let hs = &*s;
    if hs.ptr.is_null() || hs.len == 0 {
        return String::new();
    }
    let bytes = std::slice::from_raw_parts(hs.ptr as *const u8, hs.len as usize);
    String::from_utf8_lossy(bytes).into_owned()
}

// ============================================================================
// Plugin method table
// ============================================================================

rayzor_plugin::declare_native_methods! {
    RAYZOR_WINDOW_METHODS;
    "rayzor_window_Window", "create",           static,   "rayzor_window_create",              [Ptr, I64, I64, I64, I64, I64] => Ptr;
    "rayzor_window_Window", "createCentered",   static,   "rayzor_window_create_centered",     [Ptr, I64, I64]                => Ptr;
    "rayzor_window_Window", "pollEvents",       instance, "rayzor_window_poll_events",         [Ptr]           => I64;
    "rayzor_window_Window", "isKeyDown",        instance, "rayzor_window_is_key_down",         [Ptr, I64]      => I64;
    "rayzor_window_Window", "getHandle",        instance, "rayzor_window_get_handle",          [Ptr]           => Ptr;
    "rayzor_window_Window", "getDisplayHandle", instance, "rayzor_window_get_display_handle",  [Ptr]           => Ptr;
    "rayzor_window_Window", "getWidth",         instance, "rayzor_window_get_width",           [Ptr]           => I64;
    "rayzor_window_Window", "getHeight",        instance, "rayzor_window_get_height",          [Ptr]           => I64;
    "rayzor_window_Window", "wasResized",       instance, "rayzor_window_was_resized",         [Ptr]           => I64;
    "rayzor_window_Window", "setTitle",         instance, "rayzor_window_set_title",           [Ptr, Ptr]      => Void;
    "rayzor_window_Window", "destroy",          instance, "rayzor_window_destroy",             [Ptr]           => Void;
}

// ============================================================================
// Plugin exports — universal entry point
// ============================================================================

fn get_runtime_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("rayzor_window_create", rayzor_window_create as *const u8),
        ("rayzor_window_create_centered", rayzor_window_create_centered as *const u8),
        ("rayzor_window_poll_events", rayzor_window_poll_events as *const u8),
        ("rayzor_window_is_key_down", rayzor_window_is_key_down as *const u8),
        ("rayzor_window_get_handle", rayzor_window_get_handle as *const u8),
        ("rayzor_window_get_display_handle", rayzor_window_get_display_handle as *const u8),
        ("rayzor_window_get_width", rayzor_window_get_width as *const u8),
        ("rayzor_window_get_height", rayzor_window_get_height as *const u8),
        ("rayzor_window_was_resized", rayzor_window_was_resized as *const u8),
        ("rayzor_window_set_title", rayzor_window_set_title as *const u8),
        ("rayzor_window_destroy", rayzor_window_destroy as *const u8),
    ]
}

// Single universal entry point — no more hardcoded function name guessing
rayzor_plugin::rpkg_entry!(RAYZOR_WINDOW_METHODS, get_runtime_symbols);
