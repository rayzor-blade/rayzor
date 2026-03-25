// All extern "C" functions in this crate are FFI entry points called by the JIT runtime.
#![allow(clippy::missing_safety_doc)]

//! rayzor-window — cross-platform native windowing via raw system FFI.
//!
//! Zero third-party dependencies. Uses system APIs directly:
//! - macOS: objc_msgSend + Cocoa (libobjc.dylib)
//! - Linux: dlopen("libX11.so") + X11
//! - Windows: CreateWindowExW (user32.dll)

use std::ffi::c_void;

pub mod event;

#[cfg(target_os = "macos")]
mod cocoa;

#[cfg(target_os = "linux")]
mod x11;

#[cfg(target_os = "windows")]
mod win32;

use rayzor_runtime::haxe_string::HaxeString;

// ============================================================================
// NativeWindow — platform-agnostic wrapper
// ============================================================================

pub struct NativeWindow {
    #[cfg(target_os = "macos")]
    inner: cocoa::CocoaWindow,
    #[cfg(target_os = "linux")]
    inner: x11::X11Window,
    #[cfg(target_os = "windows")]
    inner: win32::Win32Window,
}

macro_rules! platform_create {
    ($method:ident, $($arg:expr),*) => {{
        #[cfg(target_os = "macos")]
        { match cocoa::CocoaWindow::$method($($arg),*) { Some(w) => Box::into_raw(Box::new(NativeWindow { inner: w })), None => std::ptr::null_mut() } }
        #[cfg(target_os = "linux")]
        { match x11::X11Window::$method($($arg),*) { Some(w) => Box::into_raw(Box::new(NativeWindow { inner: w })), None => std::ptr::null_mut() } }
        #[cfg(target_os = "windows")]
        { match win32::Win32Window::$method($($arg),*) { Some(w) => Box::into_raw(Box::new(NativeWindow { inner: w })), None => std::ptr::null_mut() } }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        { std::ptr::null_mut() }
    }};
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
    let t = haxe_string_to_str(title);
    platform_create!(create, &t, x, y, w, h, style)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_create_centered(
    title: *const HaxeString,
    w: i32,
    h: i32,
) -> *mut NativeWindow {
    let t = haxe_string_to_str(title);
    platform_create!(create_centered, &t, w, h)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_poll_events(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    if (*win).inner.poll_events() {
        1
    } else {
        0
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_is_key_down(win: *mut NativeWindow, key: i32) -> i32 {
    if win.is_null() {
        return 0;
    }
    if (*win).inner.is_key_down(key) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_handle(win: *mut NativeWindow) -> *mut c_void {
    if win.is_null() {
        return std::ptr::null_mut();
    }
    #[cfg(target_os = "macos")]
    {
        (*win).inner.ns_view
    }
    #[cfg(target_os = "linux")]
    {
        (*win).inner.window as *mut c_void
    }
    #[cfg(target_os = "windows")]
    {
        (*win).inner.hwnd
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        std::ptr::null_mut()
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_display_handle(win: *mut NativeWindow) -> *mut c_void {
    if win.is_null() {
        return std::ptr::null_mut();
    }
    #[cfg(target_os = "linux")]
    {
        (*win).inner.display
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_width(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win).inner.width as i32
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_height(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win).inner.height as i32
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_x(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win).inner.get_position().0
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_y(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win).inner.get_position().1
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_was_resized(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    if (*win).inner.resized {
        1
    } else {
        0
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_position(win: *mut NativeWindow, x: i32, y: i32) {
    if !win.is_null() {
        (*win).inner.set_position(x, y);
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_size(win: *mut NativeWindow, w: i32, h: i32) {
    if !win.is_null() {
        (*win).inner.set_size(w, h);
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_min_size(win: *mut NativeWindow, w: i32, h: i32) {
    if !win.is_null() {
        (*win).inner.set_min_size(w, h);
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_max_size(win: *mut NativeWindow, w: i32, h: i32) {
    if !win.is_null() {
        (*win).inner.set_max_size(w, h);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_title(win: *mut NativeWindow, title: *const HaxeString) {
    if win.is_null() {
        return;
    }
    let t = haxe_string_to_str(title);
    (*win).inner.set_title(&t);
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_fullscreen(win: *mut NativeWindow, fs: i32) {
    if !win.is_null() {
        (*win).inner.set_fullscreen(fs != 0);
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_visible(win: *mut NativeWindow, vis: i32) {
    if !win.is_null() {
        (*win).inner.set_visible(vis != 0);
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_floating(win: *mut NativeWindow, on_top: i32) {
    if !win.is_null() {
        (*win).inner.set_floating(on_top != 0);
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_set_opacity(win: *mut NativeWindow, opacity: f64) {
    if !win.is_null() {
        (*win).inner.set_opacity(opacity);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_is_fullscreen(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    if (*win).inner.is_fullscreen() {
        1
    } else {
        0
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_is_visible(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    if (*win).inner.is_visible() {
        1
    } else {
        0
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_is_minimized(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    if (*win).inner.is_minimized() {
        1
    } else {
        0
    }
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_is_focused(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    if (*win).inner.is_focused() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_mouse_x(win: *mut NativeWindow) -> f64 {
    if win.is_null() {
        return 0.0;
    }
    (*win).inner.get_mouse_x()
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_get_mouse_y(win: *mut NativeWindow) -> f64 {
    if win.is_null() {
        return 0.0;
    }
    (*win).inner.get_mouse_y()
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_is_mouse_down(win: *mut NativeWindow, button: i32) -> i32 {
    if win.is_null() {
        return 0;
    }
    if (*win).inner.is_mouse_down(button) {
        1
    } else {
        0
    }
}

// --- Event Queue ---

#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_count(win: *mut NativeWindow) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win).inner.events.len() as i32
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_type(win: *mut NativeWindow, index: i32) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.event_type)
        .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_x(win: *mut NativeWindow, index: i32) -> f64 {
    if win.is_null() {
        return 0.0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.x)
        .unwrap_or(0.0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_y(win: *mut NativeWindow, index: i32) -> f64 {
    if win.is_null() {
        return 0.0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.y)
        .unwrap_or(0.0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_key(win: *mut NativeWindow, index: i32) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.key)
        .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_button(win: *mut NativeWindow, index: i32) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.button)
        .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_modifiers(win: *mut NativeWindow, index: i32) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.modifiers)
        .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_width(win: *mut NativeWindow, index: i32) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.width)
        .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_height(win: *mut NativeWindow, index: i32) -> i32 {
    if win.is_null() {
        return 0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.height)
        .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_scroll_x(win: *mut NativeWindow, index: i32) -> f64 {
    if win.is_null() {
        return 0.0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.scroll_x)
        .unwrap_or(0.0)
}
#[no_mangle]
pub unsafe extern "C" fn rayzor_window_event_scroll_y(win: *mut NativeWindow, index: i32) -> f64 {
    if win.is_null() {
        return 0.0;
    }
    (*win)
        .inner
        .events
        .get(index as usize)
        .map(|e| e.scroll_y)
        .unwrap_or(0.0)
}

// --- Cleanup ---

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
    let bytes = std::slice::from_raw_parts(hs.ptr as *const u8, hs.len);
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
    "rayzor_window_Window", "getX",             instance, "rayzor_window_get_x",               [Ptr]           => I64;
    "rayzor_window_Window", "getY",             instance, "rayzor_window_get_y",               [Ptr]           => I64;
    "rayzor_window_Window", "wasResized",       instance, "rayzor_window_was_resized",         [Ptr]           => I64;
    "rayzor_window_Window", "setPosition",      instance, "rayzor_window_set_position",        [Ptr, I64, I64] => Void;
    "rayzor_window_Window", "setSize",          instance, "rayzor_window_set_size",            [Ptr, I64, I64] => Void;
    "rayzor_window_Window", "setMinSize",       instance, "rayzor_window_set_min_size",        [Ptr, I64, I64] => Void;
    "rayzor_window_Window", "setMaxSize",       instance, "rayzor_window_set_max_size",        [Ptr, I64, I64] => Void;
    "rayzor_window_Window", "setTitle",         instance, "rayzor_window_set_title",           [Ptr, Ptr]      => Void;
    "rayzor_window_Window", "setFullscreen",    instance, "rayzor_window_set_fullscreen",      [Ptr, I64]      => Void;
    "rayzor_window_Window", "setVisible",       instance, "rayzor_window_set_visible",         [Ptr, I64]      => Void;
    "rayzor_window_Window", "setFloating",      instance, "rayzor_window_set_floating",        [Ptr, I64]      => Void;
    "rayzor_window_Window", "setOpacity",       instance, "rayzor_window_set_opacity",         [Ptr, F64]      => Void;
    "rayzor_window_Window", "isFullscreen",     instance, "rayzor_window_is_fullscreen",       [Ptr]           => I64;
    "rayzor_window_Window", "isVisible",        instance, "rayzor_window_is_visible",          [Ptr]           => I64;
    "rayzor_window_Window", "isMinimized",      instance, "rayzor_window_is_minimized",        [Ptr]           => I64;
    "rayzor_window_Window", "isFocused",        instance, "rayzor_window_is_focused",          [Ptr]           => I64;
    "rayzor_window_Window", "getMouseX",        instance, "rayzor_window_get_mouse_x",         [Ptr]           => F64;
    "rayzor_window_Window", "getMouseY",        instance, "rayzor_window_get_mouse_y",         [Ptr]           => F64;
    "rayzor_window_Window", "isMouseDown",      instance, "rayzor_window_is_mouse_down",       [Ptr, I64]      => I64;
    // Event queue
    "rayzor_window_Window", "eventCount",       instance, "rayzor_window_event_count",         [Ptr]           => I64;
    "rayzor_window_Window", "eventType",        instance, "rayzor_window_event_type",          [Ptr, I64]      => I64;
    "rayzor_window_Window", "eventX",           instance, "rayzor_window_event_x",             [Ptr, I64]      => F64;
    "rayzor_window_Window", "eventY",           instance, "rayzor_window_event_y",             [Ptr, I64]      => F64;
    "rayzor_window_Window", "eventKey",         instance, "rayzor_window_event_key",           [Ptr, I64]      => I64;
    "rayzor_window_Window", "eventButton",      instance, "rayzor_window_event_button",        [Ptr, I64]      => I64;
    "rayzor_window_Window", "eventModifiers",   instance, "rayzor_window_event_modifiers",     [Ptr, I64]      => I64;
    "rayzor_window_Window", "eventWidth",       instance, "rayzor_window_event_width",         [Ptr, I64]      => I64;
    "rayzor_window_Window", "eventHeight",      instance, "rayzor_window_event_height",        [Ptr, I64]      => I64;
    "rayzor_window_Window", "eventScrollX",     instance, "rayzor_window_event_scroll_x",      [Ptr, I64]      => F64;
    "rayzor_window_Window", "eventScrollY",     instance, "rayzor_window_event_scroll_y",      [Ptr, I64]      => F64;
    // Cleanup
    "rayzor_window_Window", "destroy",          instance, "rayzor_window_destroy",             [Ptr]           => Void;
}

fn get_runtime_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("rayzor_window_create", rayzor_window_create as *const u8),
        (
            "rayzor_window_create_centered",
            rayzor_window_create_centered as *const u8,
        ),
        (
            "rayzor_window_poll_events",
            rayzor_window_poll_events as *const u8,
        ),
        (
            "rayzor_window_is_key_down",
            rayzor_window_is_key_down as *const u8,
        ),
        (
            "rayzor_window_get_handle",
            rayzor_window_get_handle as *const u8,
        ),
        (
            "rayzor_window_get_display_handle",
            rayzor_window_get_display_handle as *const u8,
        ),
        (
            "rayzor_window_get_width",
            rayzor_window_get_width as *const u8,
        ),
        (
            "rayzor_window_get_height",
            rayzor_window_get_height as *const u8,
        ),
        ("rayzor_window_get_x", rayzor_window_get_x as *const u8),
        ("rayzor_window_get_y", rayzor_window_get_y as *const u8),
        (
            "rayzor_window_was_resized",
            rayzor_window_was_resized as *const u8,
        ),
        (
            "rayzor_window_set_position",
            rayzor_window_set_position as *const u8,
        ),
        (
            "rayzor_window_set_size",
            rayzor_window_set_size as *const u8,
        ),
        (
            "rayzor_window_set_min_size",
            rayzor_window_set_min_size as *const u8,
        ),
        (
            "rayzor_window_set_max_size",
            rayzor_window_set_max_size as *const u8,
        ),
        (
            "rayzor_window_set_title",
            rayzor_window_set_title as *const u8,
        ),
        (
            "rayzor_window_set_fullscreen",
            rayzor_window_set_fullscreen as *const u8,
        ),
        (
            "rayzor_window_set_visible",
            rayzor_window_set_visible as *const u8,
        ),
        (
            "rayzor_window_set_floating",
            rayzor_window_set_floating as *const u8,
        ),
        (
            "rayzor_window_set_opacity",
            rayzor_window_set_opacity as *const u8,
        ),
        (
            "rayzor_window_is_fullscreen",
            rayzor_window_is_fullscreen as *const u8,
        ),
        (
            "rayzor_window_is_visible",
            rayzor_window_is_visible as *const u8,
        ),
        (
            "rayzor_window_is_minimized",
            rayzor_window_is_minimized as *const u8,
        ),
        (
            "rayzor_window_is_focused",
            rayzor_window_is_focused as *const u8,
        ),
        (
            "rayzor_window_get_mouse_x",
            rayzor_window_get_mouse_x as *const u8,
        ),
        (
            "rayzor_window_get_mouse_y",
            rayzor_window_get_mouse_y as *const u8,
        ),
        (
            "rayzor_window_is_mouse_down",
            rayzor_window_is_mouse_down as *const u8,
        ),
        // Event queue
        (
            "rayzor_window_event_count",
            rayzor_window_event_count as *const u8,
        ),
        (
            "rayzor_window_event_type",
            rayzor_window_event_type as *const u8,
        ),
        ("rayzor_window_event_x", rayzor_window_event_x as *const u8),
        ("rayzor_window_event_y", rayzor_window_event_y as *const u8),
        (
            "rayzor_window_event_key",
            rayzor_window_event_key as *const u8,
        ),
        (
            "rayzor_window_event_button",
            rayzor_window_event_button as *const u8,
        ),
        (
            "rayzor_window_event_modifiers",
            rayzor_window_event_modifiers as *const u8,
        ),
        (
            "rayzor_window_event_width",
            rayzor_window_event_width as *const u8,
        ),
        (
            "rayzor_window_event_height",
            rayzor_window_event_height as *const u8,
        ),
        (
            "rayzor_window_event_scroll_x",
            rayzor_window_event_scroll_x as *const u8,
        ),
        (
            "rayzor_window_event_scroll_y",
            rayzor_window_event_scroll_y as *const u8,
        ),
        ("rayzor_window_destroy", rayzor_window_destroy as *const u8),
    ]
}

rayzor_plugin::rpkg_entry!(RAYZOR_WINDOW_METHODS, get_runtime_symbols);
