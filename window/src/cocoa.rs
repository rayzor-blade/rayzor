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
type MsgSendInitWindow = unsafe extern "C" fn(Id, Sel, CGRect, NSUInteger, NSUInteger, i32) -> Id;
type MsgSendEvent = unsafe extern "C" fn(Id, Sel, NSUInteger, Id, Id, i32) -> Id;
type MsgSendBool = unsafe extern "C" fn(Id, Sel) -> i8;
type MsgSendU16 = unsafe extern "C" fn(Id, Sel) -> u16;
type MsgSendNSUInt = unsafe extern "C" fn(Id, Sel) -> NSUInteger;
type MsgSendRect = unsafe extern "C" fn(Id, Sel) -> CGRect;

fn msg_fn() -> *const c_void {
    objc_msgSend as *const c_void
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
        let send0: MsgSend0 = std::mem::transmute(msg_fn());
        let send_rect: MsgSendRect = std::mem::transmute(msg_fn());

        let screen = send0(cls("NSScreen") as Id, sel("mainScreen"));
        let screen_frame = send_rect(screen, sel("frame"));

        let x = ((screen_frame.width - w as f64) / 2.0) as i32;
        let y = ((screen_frame.height - h as f64) / 2.0) as i32;

        Self::create(title, x, y, w, h, 1 | 2 | 4 | 8 | 16)
    }

    pub unsafe fn poll_events(&mut self) -> bool {
        self.resized = false;

        let send0: MsgSend0 = std::mem::transmute(msg_fn());
        let send1str: MsgSend1Str = std::mem::transmute(msg_fn());
        let send1ptr: MsgSend1Ptr = std::mem::transmute(msg_fn());
        let send_event: MsgSendEvent = std::mem::transmute(msg_fn());
        let send_bool: MsgSendBool = std::mem::transmute(msg_fn());
        let send_nsuint: MsgSendNSUInt = std::mem::transmute(msg_fn());
        let send_u16: MsgSendU16 = std::mem::transmute(msg_fn());

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

            // Track key events (10 = keyDown, 11 = keyUp)
            let event_type = send_nsuint(event, sel("type"));
            if event_type == 10 || event_type == 11 {
                let keycode = send_u16(event, sel("keyCode"));
                let vk = cocoa_keycode_to_key(keycode);
                if vk < 256 { self.key_states[vk] = event_type == 10; }
            }

            send1ptr(app, sel("sendEvent:"), event);
        }

        send_bool(self.ns_window, sel("isVisible")) != 0
    }

    pub fn is_key_down(&self, key: i32) -> bool {
        (key >= 0 && key < 256) && self.key_states[key as usize]
    }

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
