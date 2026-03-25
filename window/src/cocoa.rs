//! macOS window implementation via raw Cocoa FFI.
//!
//! Uses objc_msgSend directly — no third-party Objective-C bindings.
//! libobjc.dylib is always available on macOS.

use std::ffi::c_void;
use std::os::raw::c_char;

// Objective-C runtime types
type Id = *mut c_void;
type Sel = *mut c_void;
type Class = *mut c_void;
type NSUInteger = usize;
type CGFloat = f64;
type BOOL = i8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CGRect {
    pub x: CGFloat,
    pub y: CGFloat,
    pub width: CGFloat,
    pub height: CGFloat,
}

// Raw Objective-C runtime FFI — always in libobjc.dylib
extern "C" {
    fn objc_getClass(name: *const c_char) -> Class;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend(receiver: Id, sel: Sel, ...) -> Id;
}

/// Helper macros for objc_msgSend with proper casts
macro_rules! msg {
    ($obj:expr, $sel:expr) => {
        objc_msgSend($obj, sel_registerName(concat!($sel, "\0").as_ptr() as *const c_char))
    };
    ($obj:expr, $sel:expr, $($arg:expr),+) => {
        objc_msgSend($obj, sel_registerName(concat!($sel, "\0").as_ptr() as *const c_char), $($arg),+)
    };
}

macro_rules! cls {
    ($name:expr) => {
        objc_getClass(concat!($name, "\0").as_ptr() as *const c_char)
    };
}

pub struct CocoaWindow {
    pub ns_window: Id,
    pub ns_view: Id,
    pub width: u32,
    pub height: u32,
    pub resized: bool,
    pub should_close: bool,
    key_states: [bool; 256],
}

impl CocoaWindow {
    pub unsafe fn create(title: &str, x: i32, y: i32, w: i32, h: i32, style: i32) -> Option<Self> {
        eprintln!("[cocoa] create: {}x{} at ({},{}) style={}", w, h, x, y, style);
        // [NSApplication sharedApplication]
        let cls_ptr = objc_getClass("NSApplication\0".as_ptr() as *const c_char);
        eprintln!("[cocoa] NSApplication class={:?}", cls_ptr);
        let sel_ptr = sel_registerName("sharedApplication\0".as_ptr() as *const c_char);
        eprintln!("[cocoa] sharedApplication sel={:?}", sel_ptr);
        let app: Id = objc_msgSend(cls_ptr, sel_ptr);
        eprintln!("[cocoa] app={:?}", app);
        if app.is_null() {
            eprintln!("[cocoa] NSApplication.sharedApplication returned null!");
            return None;
        }

        // [app setActivationPolicy:NSApplicationActivationPolicyRegular]
        let _: Id = msg!(app, "setActivationPolicy:", 0 as NSUInteger);

        // Convert style flags to NSWindowStyleMask
        let mask = style_to_cocoa_mask(style);

        // [[NSWindow alloc] initWithContentRect:styleMask:backing:defer:]
        let frame = CGRect {
            x: x as CGFloat,
            y: y as CGFloat,
            width: w as CGFloat,
            height: h as CGFloat,
        };
        let alloc: Id = msg!(cls!("NSWindow"), "alloc");
        let window: Id = {
            type MsgSendInit = unsafe extern "C" fn(Id, Sel, CGRect, NSUInteger, NSUInteger, BOOL) -> Id;
            let f: MsgSendInit = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                alloc,
                sel_registerName("initWithContentRect:styleMask:backing:defer:\0".as_ptr() as *const c_char),
                frame,
                mask,
                2, // NSBackingStoreBuffered
                0, // defer: NO
            )
        };

        if window.is_null() {
            return None;
        }

        // [window setTitle:@"..."]
        let title_cstr = std::ffi::CString::new(title).ok()?;
        let ns_title: Id = msg!(cls!("NSString"), "stringWithUTF8String:", title_cstr.as_ptr());
        let _: Id = msg!(window, "setTitle:", ns_title);

        // [window makeKeyAndOrderFront:nil]
        let _: Id = msg!(window, "makeKeyAndOrderFront:", std::ptr::null_mut::<c_void>());

        // [app activateIgnoringOtherApps:YES]
        let _: Id = msg!(app, "activateIgnoringOtherApps:", 1i32);

        // Get content view
        let view: Id = msg!(window, "contentView");

        // [view setWantsLayer:YES] — required for Metal/wgpu surface
        let _: Id = msg!(view, "setWantsLayer:", 1i32);

        Some(CocoaWindow {
            ns_window: window,
            ns_view: view,
            width: w as u32,
            height: h as u32,
            resized: false,
            should_close: false,
            key_states: [false; 256],
        })
    }

    pub unsafe fn create_centered(title: &str, w: i32, h: i32) -> Option<Self> {
        // Get screen size for centering
        let screen: Id = msg!(cls!("NSScreen"), "mainScreen");
        let screen_frame: CGRect = {
            type MsgSendRect = unsafe extern "C" fn(Id, Sel) -> CGRect;
            let f: MsgSendRect = std::mem::transmute(objc_msgSend as *const c_void);
            f(screen, sel_registerName("frame\0".as_ptr() as *const c_char))
        };
        let x = ((screen_frame.width - w as f64) / 2.0) as i32;
        let y = ((screen_frame.height - h as f64) / 2.0) as i32;

        // Default style: titled + closable + resizable + miniaturizable + maximizable
        Self::create(title, x, y, w, h, 1 | 2 | 4 | 8 | 16)
    }

    pub unsafe fn poll_events(&mut self) -> bool {
        self.resized = false;

        let app: Id = msg!(cls!("NSApplication"), "sharedApplication");
        let mode: Id = msg!(
            cls!("NSString"),
            "stringWithUTF8String:",
            "kCFRunLoopDefaultMode\0".as_ptr() as *const c_char
        );

        loop {
            let event: Id = {
                type MsgSendEvent = unsafe extern "C" fn(Id, Sel, NSUInteger, Id, Id, BOOL) -> Id;
                let f: MsgSendEvent = std::mem::transmute(objc_msgSend as *const c_void);
                f(
                    app,
                    sel_registerName("nextEventMatchingMask:untilDate:inMode:dequeue:\0".as_ptr() as *const c_char),
                    NSUInteger::MAX,
                    std::ptr::null_mut(),
                    mode,
                    1, // dequeue: YES
                )
            };

            if event.is_null() {
                break;
            }

            // Get event type
            let event_type: NSUInteger = {
                type MsgSendInt = unsafe extern "C" fn(Id, Sel) -> NSUInteger;
                let f: MsgSendInt = std::mem::transmute(objc_msgSend as *const c_void);
                f(event, sel_registerName("type\0".as_ptr() as *const c_char))
            };

            // Track key events (10 = keyDown, 11 = keyUp)
            if event_type == 10 || event_type == 11 {
                let keycode: u16 = {
                    type MsgSendU16 = unsafe extern "C" fn(Id, Sel) -> u16;
                    let f: MsgSendU16 = std::mem::transmute(objc_msgSend as *const c_void);
                    f(event, sel_registerName("keyCode\0".as_ptr() as *const c_char))
                };
                let virtual_key = cocoa_keycode_to_key(keycode);
                if virtual_key < 256 {
                    self.key_states[virtual_key] = event_type == 10;
                }
            }

            let _: Id = msg!(app, "sendEvent:", event);
        }

        // Check if window is still visible
        let visible: BOOL = {
            type MsgSendBool = unsafe extern "C" fn(Id, Sel) -> BOOL;
            let f: MsgSendBool = std::mem::transmute(objc_msgSend as *const c_void);
            f(self.ns_window, sel_registerName("isVisible\0".as_ptr() as *const c_char))
        };

        visible != 0
    }

    pub fn is_key_down(&self, key: i32) -> bool {
        if key >= 0 && key < 256 {
            self.key_states[key as usize]
        } else {
            false
        }
    }

    pub unsafe fn set_title(&self, title: &str) {
        if let Ok(cstr) = std::ffi::CString::new(title) {
            let ns_title: Id = msg!(cls!("NSString"), "stringWithUTF8String:", cstr.as_ptr());
            let _: Id = msg!(self.ns_window, "setTitle:", ns_title);
        }
    }

    pub unsafe fn destroy(&self) {
        if !self.ns_window.is_null() {
            let _: Id = msg!(self.ns_window, "close");
        }
    }
}

fn style_to_cocoa_mask(style: i32) -> NSUInteger {
    let mut mask: NSUInteger = 0;
    if style & 1 != 0 { mask |= 1; }       // titled
    if style & 2 != 0 { mask |= 2; }       // closable
    if style & 4 != 0 { mask |= 8; }       // resizable
    if style & 8 != 0 { mask |= 4; }       // miniaturizable
    // maximizable is implicit with resizable on macOS
    if style & 32 != 0 { mask = 0; }       // frameless (borderless)
    mask
}

/// Convert macOS virtual key code to cross-platform Key code
fn cocoa_keycode_to_key(keycode: u16) -> usize {
    match keycode {
        53 => 27,   // Escape
        49 => 32,   // Space
        36 => 13,   // Return
        48 => 9,    // Tab
        51 => 8,    // Delete (backspace)
        123 => 37,  // Left
        126 => 38,  // Up
        124 => 39,  // Right
        125 => 40,  // Down
        // A-Z (macOS keycodes 0-5 = ASDWQE, etc.)
        0 => 65,    // A
        11 => 66,   // B
        8 => 67,    // C
        2 => 68,    // D
        14 => 69,   // E
        3 => 70,    // F
        5 => 71,    // G
        4 => 72,    // H
        34 => 73,   // I
        38 => 74,   // J
        40 => 75,   // K
        37 => 76,   // L
        46 => 77,   // M
        45 => 78,   // N
        31 => 79,   // O
        35 => 80,   // P
        12 => 81,   // Q
        15 => 82,   // R
        1 => 83,    // S
        17 => 84,   // T
        32 => 85,   // U
        9 => 86,    // V
        13 => 87,   // W
        7 => 88,    // X
        16 => 89,   // Y
        6 => 90,    // Z
        // F1-F12
        122 => 112, // F1
        120 => 113, // F2
        99 => 114,  // F3
        118 => 115, // F4
        96 => 116,  // F5
        97 => 117,  // F6
        98 => 118,  // F7
        100 => 119, // F8
        101 => 120, // F9
        109 => 121, // F10
        103 => 122, // F11
        111 => 123, // F12
        _ => 0,
    }
}
