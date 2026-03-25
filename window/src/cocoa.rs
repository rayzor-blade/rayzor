//! macOS window implementation via raw Cocoa FFI.
//!
//! Uses objc_msgSend directly — no third-party Objective-C bindings.
//! All calls use typed function pointers (not variadic) for correct ARM64 ABI.

use std::ffi::c_void;
use std::os::raw::c_char;

type Id = *mut c_void;
type Sel = *mut c_void;
type Class = *mut c_void;
type NSUInteger = usize;
type CGFloat = f64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CGRect {
    pub x: CGFloat,
    pub y: CGFloat,
    pub width: CGFloat,
    pub height: CGFloat,
}

extern "C" {
    fn objc_getClass(name: *const c_char) -> Class;
    fn sel_registerName(name: *const c_char) -> Sel;
    // NOTE: We never call objc_msgSend directly as variadic.
    // All calls go through typed function pointer casts.
    fn objc_msgSend() -> Id;
}

fn sel(name: &str) -> Sel {
    unsafe { sel_registerName(format!("{}\0", name).as_ptr() as *const c_char) }
}

fn cls(name: &str) -> Class {
    unsafe { objc_getClass(format!("{}\0", name).as_ptr() as *const c_char) }
}

// Typed objc_msgSend casts — ARM64 ABI requires non-variadic prototypes
type MsgSend0 = unsafe extern "C" fn(Id, Sel) -> Id;
type MsgSend1Ptr = unsafe extern "C" fn(Id, Sel, Id) -> Id;
type MsgSend1Int = unsafe extern "C" fn(Id, Sel, NSUInteger) -> Id;
type MsgSend1Str = unsafe extern "C" fn(Id, Sel, *const c_char) -> Id;
type MsgSend1Bool = unsafe extern "C" fn(Id, Sel, i32) -> Id;
type MsgSend1NSUInt = unsafe extern "C" fn(Id, Sel, NSUInteger) -> Id;
type MsgSend1CGFloat = unsafe extern "C" fn(Id, Sel, CGFloat) -> Id;
type MsgSendCGSize = unsafe extern "C" fn(Id, Sel, CGFloat, CGFloat) -> Id;
type MsgSendRect1Bool = unsafe extern "C" fn(Id, Sel, CGRect, i32) -> Id;
type MsgSendInitWindow = unsafe extern "C" fn(Id, Sel, CGRect, NSUInteger, NSUInteger, i32) -> Id;
type MsgSendEvent = unsafe extern "C" fn(Id, Sel, NSUInteger, Id, Id, i32) -> Id;
type MsgSendBool = unsafe extern "C" fn(Id, Sel) -> i8;
type MsgSendU16 = unsafe extern "C" fn(Id, Sel) -> u16;
type MsgSendNSUInt = unsafe extern "C" fn(Id, Sel) -> NSUInteger;
type MsgSendRect = unsafe extern "C" fn(Id, Sel) -> CGRect;
type MsgSendNSPoint = unsafe extern "C" fn(Id, Sel) -> [f64; 2];
type MsgSendCGFloat = unsafe extern "C" fn(Id, Sel) -> CGFloat;

fn msg_fn() -> *const c_void {
    objc_msgSend as *const c_void
}

/// NSFullScreenWindowMask = 1 << 14
const NS_FULLSCREEN_WINDOW_MASK: NSUInteger = 1 << 14;
/// NSFloatingWindowLevel = 3
const NS_FLOATING_WINDOW_LEVEL: NSUInteger = 3;
/// NSNormalWindowLevel = 0
const NS_NORMAL_WINDOW_LEVEL: NSUInteger = 0;

pub struct CocoaWindow {
    pub ns_window: Id,
    pub ns_view: Id,
    pub width: u32,
    pub height: u32,
    pub resized: bool,
    pub should_close: bool,
    key_states: [bool; 256],
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub mouse_buttons: [bool; 5],
}

impl CocoaWindow {
    pub unsafe fn create(title: &str, x: i32, y: i32, w: i32, h: i32, style: i32) -> Option<Self> {
        let send0: MsgSend0 = std::mem::transmute(msg_fn());
        let send1ptr: MsgSend1Ptr = std::mem::transmute(msg_fn());
        let send1int: MsgSend1Int = std::mem::transmute(msg_fn());
        let send1str: MsgSend1Str = std::mem::transmute(msg_fn());
        let send1bool: MsgSend1Bool = std::mem::transmute(msg_fn());
        let send_init: MsgSendInitWindow = std::mem::transmute(msg_fn());

        // [NSApplication sharedApplication]
        let app = send0(cls("NSApplication") as Id, sel("sharedApplication"));
        if app.is_null() { return None; }

        // [app setActivationPolicy:0]
        send1int(app, sel("setActivationPolicy:"), 0);

        let mask = style_to_cocoa_mask(style);
        let frame = CGRect {
            x: x as CGFloat,
            y: y as CGFloat,
            width: w as CGFloat,
            height: h as CGFloat,
        };

        // [[NSWindow alloc] initWithContentRect:styleMask:backing:defer:]
        let alloc = send0(cls("NSWindow") as Id, sel("alloc"));
        let window = send_init(
            alloc,
            sel("initWithContentRect:styleMask:backing:defer:"),
            frame,
            mask,
            2, // NSBackingStoreBuffered
            0, // defer: NO
        );
        if window.is_null() { return None; }

        // [window setTitle:@"..."]
        let title_cstr = std::ffi::CString::new(title).ok()?;
        let ns_title = send1str(cls("NSString") as Id, sel("stringWithUTF8String:"), title_cstr.as_ptr());
        send1ptr(window, sel("setTitle:"), ns_title);

        // [window makeKeyAndOrderFront:nil]
        send1ptr(window, sel("makeKeyAndOrderFront:"), std::ptr::null_mut());

        // [app activateIgnoringOtherApps:YES]
        send1bool(app, sel("activateIgnoringOtherApps:"), 1);

        // Get content view + enable layer
        let view = send0(window, sel("contentView"));
        send1bool(view, sel("setWantsLayer:"), 1);

        // [window setAcceptsMouseMovedEvents:YES]
        send1bool(window, sel("setAcceptsMouseMovedEvents:"), 1);

        Some(CocoaWindow {
            ns_window: window,
            ns_view: view,
            width: w as u32,
            height: h as u32,
            resized: false,
            should_close: false,
            key_states: [false; 256],
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_buttons: [false; 5],
        })
    }

    pub unsafe fn create_centered(title: &str, w: i32, h: i32) -> Option<Self> {
        let send0: MsgSend0 = std::mem::transmute(msg_fn());
        let send_rect: MsgSendRect = std::mem::transmute(msg_fn());

        let screen = send0(cls("NSScreen") as Id, sel("mainScreen"));
        let screen_frame = send_rect(screen, sel("frame"));

        let x = ((screen_frame.width - w as f64) / 2.0) as i32;
        let y = ((screen_frame.height - h as f64) / 2.0) as i32;

        Self::create(title, x, y, w, h, 1 | 2 | 4 | 8 | 16)
    }

    // ========================================================================
    // Event polling
    // ========================================================================

    pub unsafe fn poll_events(&mut self) -> bool {
        self.resized = false;

        let send0: MsgSend0 = std::mem::transmute(msg_fn());
        let send1str: MsgSend1Str = std::mem::transmute(msg_fn());
        let send1ptr: MsgSend1Ptr = std::mem::transmute(msg_fn());
        let send_event: MsgSendEvent = std::mem::transmute(msg_fn());
        let send_bool: MsgSendBool = std::mem::transmute(msg_fn());
        let send_nsuint: MsgSendNSUInt = std::mem::transmute(msg_fn());
        let send_u16: MsgSendU16 = std::mem::transmute(msg_fn());
        let send_rect: MsgSendRect = std::mem::transmute(msg_fn());
        let send_nspoint: MsgSendNSPoint = std::mem::transmute(msg_fn());

        let app = send0(cls("NSApplication") as Id, sel("sharedApplication"));
        let mode = send1str(
            cls("NSString") as Id,
            sel("stringWithUTF8String:"),
            "kCFRunLoopDefaultMode\0".as_ptr() as *const c_char,
        );

        loop {
            let event = send_event(
                app,
                sel("nextEventMatchingMask:untilDate:inMode:dequeue:"),
                NSUInteger::MAX,
                std::ptr::null_mut(),
                mode,
                1, // dequeue: YES
            );
            if event.is_null() { break; }

            let event_type = send_nsuint(event, sel("type"));

            // Track key events (10 = keyDown, 11 = keyUp)
            if event_type == 10 || event_type == 11 {
                let keycode = send_u16(event, sel("keyCode"));
                let vk = cocoa_keycode_to_key(keycode);
                if vk < 256 { self.key_states[vk] = event_type == 10; }
            }

            // Track mouse button events
            match event_type {
                // 1 = NSLeftMouseDown
                1 => {
                    self.mouse_buttons[0] = true;
                    let loc = send_nspoint(event, sel("locationInWindow"));
                    self.mouse_x = loc[0];
                    self.mouse_y = loc[1];
                }
                // 2 = NSLeftMouseUp
                2 => {
                    self.mouse_buttons[0] = false;
                    let loc = send_nspoint(event, sel("locationInWindow"));
                    self.mouse_x = loc[0];
                    self.mouse_y = loc[1];
                }
                // 3 = NSRightMouseDown
                3 => {
                    self.mouse_buttons[1] = true;
                    let loc = send_nspoint(event, sel("locationInWindow"));
                    self.mouse_x = loc[0];
                    self.mouse_y = loc[1];
                }
                // 4 = NSRightMouseUp
                4 => {
                    self.mouse_buttons[1] = false;
                    let loc = send_nspoint(event, sel("locationInWindow"));
                    self.mouse_x = loc[0];
                    self.mouse_y = loc[1];
                }
                // 25 = NSOtherMouseDown (middle button)
                25 => {
                    self.mouse_buttons[2] = true;
                    let loc = send_nspoint(event, sel("locationInWindow"));
                    self.mouse_x = loc[0];
                    self.mouse_y = loc[1];
                }
                // 26 = NSOtherMouseUp (middle button)
                26 => {
                    self.mouse_buttons[2] = false;
                    let loc = send_nspoint(event, sel("locationInWindow"));
                    self.mouse_x = loc[0];
                    self.mouse_y = loc[1];
                }
                // 5 = NSMouseMoved, 6 = NSLeftMouseDragged, 7 = NSRightMouseDragged,
                // 27 = NSOtherMouseMoved
                5 | 6 | 7 | 27 => {
                    let loc = send_nspoint(event, sel("locationInWindow"));
                    self.mouse_x = loc[0];
                    self.mouse_y = loc[1];
                }
                _ => {}
            }

            send1ptr(app, sel("sendEvent:"), event);
        }

        // Detect window resize by comparing current frame to stored size
        let frame = send_rect(self.ns_view, sel("frame"));
        let new_w = frame.width as u32;
        let new_h = frame.height as u32;
        if new_w != self.width || new_h != self.height {
            self.width = new_w;
            self.height = new_h;
            self.resized = true;
        }

        send_bool(self.ns_window, sel("isVisible")) != 0
    }

    // ========================================================================
    // Key input
    // ========================================================================

    pub fn is_key_down(&self, key: i32) -> bool {
        (key >= 0 && key < 256) && self.key_states[key as usize]
    }

    // ========================================================================
    // Mouse input
    // ========================================================================

    pub fn get_mouse_x(&self) -> f64 {
        self.mouse_x
    }

    pub fn get_mouse_y(&self) -> f64 {
        self.mouse_y
    }

    pub fn is_mouse_down(&self, button: i32) -> bool {
        if button >= 0 && (button as usize) < self.mouse_buttons.len() {
            self.mouse_buttons[button as usize]
        } else {
            false
        }
    }

    // ========================================================================
    // Geometry
    // ========================================================================

    /// Returns (x, y) position of the window frame origin (bottom-left in Cocoa coords).
    pub unsafe fn get_position(&self) -> (i32, i32) {
        let send_rect: MsgSendRect = std::mem::transmute(msg_fn());
        let frame = send_rect(self.ns_window, sel("frame"));
        (frame.x as i32, frame.y as i32)
    }

    /// Sets the window origin to (x, y) while preserving current size.
    pub unsafe fn set_position(&self, x: i32, y: i32) {
        let send_rect: MsgSendRect = std::mem::transmute(msg_fn());
        let send_set_frame: MsgSendRect1Bool = std::mem::transmute(msg_fn());

        let frame = send_rect(self.ns_window, sel("frame"));
        let new_frame = CGRect {
            x: x as CGFloat,
            y: y as CGFloat,
            width: frame.width,
            height: frame.height,
        };
        // [window setFrame:newFrame display:YES]
        send_set_frame(self.ns_window, sel("setFrame:display:"), new_frame, 1);
    }

    /// Resizes the window frame to (w, h) while preserving current position.
    pub unsafe fn set_size(&mut self, w: i32, h: i32) {
        let send_rect: MsgSendRect = std::mem::transmute(msg_fn());
        let send_set_frame: MsgSendRect1Bool = std::mem::transmute(msg_fn());

        let frame = send_rect(self.ns_window, sel("frame"));
        let new_frame = CGRect {
            x: frame.x,
            y: frame.y,
            width: w as CGFloat,
            height: h as CGFloat,
        };
        // [window setFrame:newFrame display:YES]
        send_set_frame(self.ns_window, sel("setFrame:display:"), new_frame, 1);
        self.width = w as u32;
        self.height = h as u32;
    }

    /// Sets the minimum content size of the window.
    pub unsafe fn set_min_size(&self, w: i32, h: i32) {
        let send_cgsize: MsgSendCGSize = std::mem::transmute(msg_fn());
        // [window setContentMinSize:NSMakeSize(w, h)]
        send_cgsize(
            self.ns_window,
            sel("setContentMinSize:"),
            w as CGFloat,
            h as CGFloat,
        );
    }

    /// Sets the maximum content size of the window.
    pub unsafe fn set_max_size(&self, w: i32, h: i32) {
        let send_cgsize: MsgSendCGSize = std::mem::transmute(msg_fn());
        // [window setContentMaxSize:NSMakeSize(w, h)]
        send_cgsize(
            self.ns_window,
            sel("setContentMaxSize:"),
            w as CGFloat,
            h as CGFloat,
        );
    }

    // ========================================================================
    // Appearance
    // ========================================================================

    /// Toggles fullscreen mode. Cocoa toggleFullScreen: is a toggle, so we
    /// check current state first and only send the message if needed.
    pub unsafe fn set_fullscreen(&self, fs: bool) {
        if self.is_fullscreen() != fs {
            let send1ptr: MsgSend1Ptr = std::mem::transmute(msg_fn());
            // [window toggleFullScreen:nil]
            send1ptr(self.ns_window, sel("toggleFullScreen:"), std::ptr::null_mut());
        }
    }

    /// Shows or hides the window.
    pub unsafe fn set_visible(&self, visible: bool) {
        if visible {
            let send1ptr: MsgSend1Ptr = std::mem::transmute(msg_fn());
            // [window makeKeyAndOrderFront:nil]
            send1ptr(self.ns_window, sel("makeKeyAndOrderFront:"), std::ptr::null_mut());
        } else {
            let send1ptr: MsgSend1Ptr = std::mem::transmute(msg_fn());
            // [window orderOut:nil]
            send1ptr(self.ns_window, sel("orderOut:"), std::ptr::null_mut());
        }
    }

    /// Sets the window to float above all other windows (or restores normal level).
    pub unsafe fn set_floating(&self, on_top: bool) {
        let send1nsuint: MsgSend1NSUInt = std::mem::transmute(msg_fn());
        let level = if on_top { NS_FLOATING_WINDOW_LEVEL } else { NS_NORMAL_WINDOW_LEVEL };
        // [window setLevel:level]
        send1nsuint(self.ns_window, sel("setLevel:"), level);
    }

    /// Sets the window opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub unsafe fn set_opacity(&self, opacity: f64) {
        let send1cgfloat: MsgSend1CGFloat = std::mem::transmute(msg_fn());
        // [window setAlphaValue:opacity]
        send1cgfloat(self.ns_window, sel("setAlphaValue:"), opacity);
    }

    // ========================================================================
    // State queries
    // ========================================================================

    /// Returns true if the window is currently in fullscreen mode.
    pub unsafe fn is_fullscreen(&self) -> bool {
        let send_nsuint: MsgSendNSUInt = std::mem::transmute(msg_fn());
        let mask = send_nsuint(self.ns_window, sel("styleMask"));
        (mask & NS_FULLSCREEN_WINDOW_MASK) != 0
    }

    /// Returns true if the window is currently visible (on screen).
    pub unsafe fn is_visible(&self) -> bool {
        let send_bool: MsgSendBool = std::mem::transmute(msg_fn());
        send_bool(self.ns_window, sel("isVisible")) != 0
    }

    /// Returns true if the window is currently minimized to the dock.
    pub unsafe fn is_minimized(&self) -> bool {
        let send_bool: MsgSendBool = std::mem::transmute(msg_fn());
        send_bool(self.ns_window, sel("isMiniaturized")) != 0
    }

    /// Returns true if the window is the key window (has keyboard focus).
    pub unsafe fn is_focused(&self) -> bool {
        let send_bool: MsgSendBool = std::mem::transmute(msg_fn());
        send_bool(self.ns_window, sel("isKeyWindow")) != 0
    }

    // ========================================================================
    // Title / lifecycle
    // ========================================================================

    pub unsafe fn set_title(&self, title: &str) {
        let send1str: MsgSend1Str = std::mem::transmute(msg_fn());
        let send1ptr: MsgSend1Ptr = std::mem::transmute(msg_fn());
        if let Ok(cstr) = std::ffi::CString::new(title) {
            let ns_title = send1str(cls("NSString") as Id, sel("stringWithUTF8String:"), cstr.as_ptr());
            send1ptr(self.ns_window, sel("setTitle:"), ns_title);
        }
    }

    pub unsafe fn destroy(&self) {
        if !self.ns_window.is_null() {
            let send0: MsgSend0 = std::mem::transmute(msg_fn());
            send0(self.ns_window, sel("close"));
        }
    }
}

fn style_to_cocoa_mask(style: i32) -> NSUInteger {
    let mut mask: NSUInteger = 0;
    if style & 1 != 0 { mask |= 1; }   // titled
    if style & 2 != 0 { mask |= 2; }   // closable
    if style & 4 != 0 { mask |= 8; }   // resizable
    if style & 8 != 0 { mask |= 4; }   // miniaturizable
    if style & 32 != 0 { mask = 0; }   // frameless
    mask
}

fn cocoa_keycode_to_key(keycode: u16) -> usize {
    match keycode {
        53 => 27, 49 => 32, 36 => 13, 48 => 9, 51 => 8,
        123 => 37, 126 => 38, 124 => 39, 125 => 40,
        0 => 65, 11 => 66, 8 => 67, 2 => 68, 14 => 69, 3 => 70,
        5 => 71, 4 => 72, 34 => 73, 38 => 74, 40 => 75, 37 => 76,
        46 => 77, 45 => 78, 31 => 79, 35 => 80, 12 => 81, 15 => 82,
        1 => 83, 17 => 84, 32 => 85, 9 => 86, 13 => 87, 7 => 88,
        16 => 89, 6 => 90,
        122 => 112, 120 => 113, 99 => 114, 118 => 115,
        96 => 116, 97 => 117, 98 => 118, 100 => 119,
        101 => 120, 109 => 121, 103 => 122, 111 => 123,
        _ => 0,
    }
}
