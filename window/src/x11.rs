//! Linux window implementation via runtime-loaded X11.
//!
//! Uses dlopen("libX11.so.6") at runtime — zero build-time X11 dependency.
//! All Xlib functions are loaded via dlsym and stored as typed function pointers.

#![cfg(target_os = "linux")]

use std::ffi::{c_void, CString};
use std::os::raw::c_char;

// ============================================================================
// X11 types
// ============================================================================

type Display = *mut c_void;
type Window = u64; // XID
type Atom = u64;
type Bool = i32;
type KeySym = u64;
type Colormap = u64;

/// XEvent is a 192-byte union. We use a raw byte buffer and read fields by offset.
/// Layout (64-bit): type at offset 0 (i32).
/// We define named accessors below.
#[repr(C)]
#[derive(Copy, Clone)]
struct XEvent {
    data: [u8; 192],
}

impl XEvent {
    fn new() -> Self {
        XEvent { data: [0u8; 192] }
    }

    /// Event type — first i32 of every XEvent variant.
    fn event_type(&self) -> i32 {
        i32::from_ne_bytes([self.data[0], self.data[1], self.data[2], self.data[3]])
    }

    // KeyPress / KeyRelease: keycode at offset 84 (u32)
    fn keycode(&self) -> u32 {
        u32::from_ne_bytes([self.data[84], self.data[85], self.data[86], self.data[87]])
    }

    // ButtonPress / ButtonRelease: button at offset 84 (u32)
    fn button(&self) -> u32 {
        u32::from_ne_bytes([self.data[84], self.data[85], self.data[86], self.data[87]])
    }

    // MotionNotify: x at offset 64 (i32), y at offset 68 (i32)
    fn motion_x(&self) -> i32 {
        i32::from_ne_bytes([self.data[64], self.data[65], self.data[66], self.data[67]])
    }

    fn motion_y(&self) -> i32 {
        i32::from_ne_bytes([self.data[68], self.data[69], self.data[70], self.data[71]])
    }

    // ConfigureNotify: x at offset 32 (i32), y at offset 36 (i32),
    //                  width at offset 40 (i32), height at offset 44 (i32)
    fn configure_x(&self) -> i32 {
        i32::from_ne_bytes([self.data[32], self.data[33], self.data[34], self.data[35]])
    }

    fn configure_y(&self) -> i32 {
        i32::from_ne_bytes([self.data[36], self.data[37], self.data[38], self.data[39]])
    }

    fn configure_width(&self) -> i32 {
        i32::from_ne_bytes([self.data[40], self.data[41], self.data[42], self.data[43]])
    }

    fn configure_height(&self) -> i32 {
        i32::from_ne_bytes([self.data[44], self.data[45], self.data[46], self.data[47]])
    }

    // ClientMessage: message_type (Atom) at offset 40 (u64 on 64-bit),
    //                data.l[0] (long) at offset 56
    fn client_message_type(&self) -> Atom {
        u64::from_ne_bytes([
            self.data[40],
            self.data[41],
            self.data[42],
            self.data[43],
            self.data[44],
            self.data[45],
            self.data[46],
            self.data[47],
        ])
    }

    fn client_data_l0(&self) -> u64 {
        u64::from_ne_bytes([
            self.data[56],
            self.data[57],
            self.data[58],
            self.data[59],
            self.data[60],
            self.data[61],
            self.data[62],
            self.data[63],
        ])
    }
}

// ============================================================================
// X11 event type constants
// ============================================================================

const KEY_PRESS: i32 = 2;
const KEY_RELEASE: i32 = 3;
const BUTTON_PRESS: i32 = 4;
const BUTTON_RELEASE: i32 = 5;
const MOTION_NOTIFY: i32 = 6;
const EXPOSE: i32 = 12;
const CONFIGURE_NOTIFY: i32 = 22;
const CLIENT_MESSAGE: i32 = 33;

// X11 event masks
const KEY_PRESS_MASK: i64 = 1 << 0;
const KEY_RELEASE_MASK: i64 = 1 << 1;
const BUTTON_PRESS_MASK: i64 = 1 << 2;
const BUTTON_RELEASE_MASK: i64 = 1 << 3;
const POINTER_MOTION_MASK: i64 = 1 << 6;
const EXPOSURE_MASK: i64 = 1 << 15;
const STRUCTURE_NOTIFY_MASK: i64 = 1 << 17;

const EVENT_MASK: i64 = KEY_PRESS_MASK
    | KEY_RELEASE_MASK
    | BUTTON_PRESS_MASK
    | BUTTON_RELEASE_MASK
    | POINTER_MOTION_MASK
    | EXPOSURE_MASK
    | STRUCTURE_NOTIFY_MASK;

// ============================================================================
// X11 function pointer table — loaded once via dlopen/dlsym
// ============================================================================

#[allow(non_snake_case)]
struct X11Lib {
    _lib: *mut c_void,
    XOpenDisplay: unsafe extern "C" fn(*const c_char) -> Display,
    XCloseDisplay: unsafe extern "C" fn(Display) -> i32,
    XCreateSimpleWindow:
        unsafe extern "C" fn(Display, Window, i32, i32, u32, u32, u32, u64, u64) -> Window,
    XMapWindow: unsafe extern "C" fn(Display, Window) -> i32,
    XUnmapWindow: unsafe extern "C" fn(Display, Window) -> i32,
    XDestroyWindow: unsafe extern "C" fn(Display, Window) -> i32,
    XStoreName: unsafe extern "C" fn(Display, Window, *const c_char) -> i32,
    XMoveResizeWindow: unsafe extern "C" fn(Display, Window, i32, i32, u32, u32) -> i32,
    XMoveWindow: unsafe extern "C" fn(Display, Window, i32, i32) -> i32,
    XResizeWindow: unsafe extern "C" fn(Display, Window, u32, u32) -> i32,
    XNextEvent: unsafe extern "C" fn(Display, *mut XEvent) -> i32,
    XPending: unsafe extern "C" fn(Display) -> i32,
    XSelectInput: unsafe extern "C" fn(Display, Window, i64) -> i32,
    XDefaultScreen: unsafe extern "C" fn(Display) -> i32,
    XRootWindow: unsafe extern "C" fn(Display, i32) -> Window,
    XDefaultGC: unsafe extern "C" fn(Display, i32) -> *mut c_void,
    XBlackPixel: unsafe extern "C" fn(Display, i32) -> u64,
    XWhitePixel: unsafe extern "C" fn(Display, i32) -> u64,
    XDisplayWidth: unsafe extern "C" fn(Display, i32) -> i32,
    XDisplayHeight: unsafe extern "C" fn(Display, i32) -> i32,
    XInternAtom: unsafe extern "C" fn(Display, *const c_char, Bool) -> Atom,
    XSetWMProtocols: unsafe extern "C" fn(Display, Window, *mut Atom, i32) -> i32,
    XFlush: unsafe extern "C" fn(Display) -> i32,
    XLookupKeysym: unsafe extern "C" fn(*mut XEvent, i32) -> KeySym,
    XGetWindowAttributes: unsafe extern "C" fn(Display, Window, *mut XWindowAttributes) -> i32,
    XDefaultColormap: unsafe extern "C" fn(Display, i32) -> Colormap,
}

/// XWindowAttributes — used to query map_state for visibility and window geometry.
#[repr(C)]
struct XWindowAttributes {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    border_width: i32,
    depth: i32,
    visual: *mut c_void,
    root: Window,
    class: i32,
    bit_gravity: i32,
    win_gravity: i32,
    backing_store: i32,
    backing_planes: u64,
    backing_pixel: u64,
    save_under: Bool,
    colormap: Colormap,
    map_installed: Bool,
    map_state: i32, // 0=Unmapped, 1=Unviewable, 2=Viewable (IsViewable)
    all_event_masks: i64,
    your_event_mask: i64,
    do_not_propagate_mask: i64,
    override_redirect: Bool,
    screen: *mut c_void,
}

impl XWindowAttributes {
    fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

static mut X11: Option<X11Lib> = None;

/// Load (or return cached) X11 function table. Returns None if libX11.so.6 is unavailable.
fn x11() -> Option<&'static X11Lib> {
    unsafe {
        if X11.is_some() {
            return X11.as_ref();
        }

        let lib = libc::dlopen(b"libX11.so.6\0".as_ptr() as *const c_char, libc::RTLD_LAZY);
        if lib.is_null() {
            // Fallback: try without version suffix
            let lib2 = libc::dlopen(b"libX11.so\0".as_ptr() as *const c_char, libc::RTLD_LAZY);
            if lib2.is_null() {
                return None;
            }
            return load_x11_symbols(lib2);
        }
        load_x11_symbols(lib)
    }
}

unsafe fn load_x11_symbols(lib: *mut c_void) -> Option<&'static X11Lib> {
    macro_rules! load_sym {
        ($name:ident, $ty:ty) => {{
            let sym = libc::dlsym(
                lib,
                concat!(stringify!($name), "\0").as_ptr() as *const c_char,
            );
            if sym.is_null() {
                eprintln!(
                    "[rayzor-window] Failed to load X11 symbol: {}",
                    stringify!($name)
                );
                return None;
            }
            std::mem::transmute::<*mut c_void, $ty>(sym)
        }};
    }

    let x11lib = X11Lib {
        _lib: lib,
        XOpenDisplay: load_sym!(XOpenDisplay, unsafe extern "C" fn(*const c_char) -> Display),
        XCloseDisplay: load_sym!(XCloseDisplay, unsafe extern "C" fn(Display) -> i32),
        XCreateSimpleWindow: load_sym!(
            XCreateSimpleWindow,
            unsafe extern "C" fn(Display, Window, i32, i32, u32, u32, u32, u64, u64) -> Window
        ),
        XMapWindow: load_sym!(XMapWindow, unsafe extern "C" fn(Display, Window) -> i32),
        XUnmapWindow: load_sym!(XUnmapWindow, unsafe extern "C" fn(Display, Window) -> i32),
        XDestroyWindow: load_sym!(XDestroyWindow, unsafe extern "C" fn(Display, Window) -> i32),
        XStoreName: load_sym!(
            XStoreName,
            unsafe extern "C" fn(Display, Window, *const c_char) -> i32
        ),
        XMoveResizeWindow: load_sym!(
            XMoveResizeWindow,
            unsafe extern "C" fn(Display, Window, i32, i32, u32, u32) -> i32
        ),
        XMoveWindow: load_sym!(
            XMoveWindow,
            unsafe extern "C" fn(Display, Window, i32, i32) -> i32
        ),
        XResizeWindow: load_sym!(
            XResizeWindow,
            unsafe extern "C" fn(Display, Window, u32, u32) -> i32
        ),
        XNextEvent: load_sym!(
            XNextEvent,
            unsafe extern "C" fn(Display, *mut XEvent) -> i32
        ),
        XPending: load_sym!(XPending, unsafe extern "C" fn(Display) -> i32),
        XSelectInput: load_sym!(
            XSelectInput,
            unsafe extern "C" fn(Display, Window, i64) -> i32
        ),
        XDefaultScreen: load_sym!(XDefaultScreen, unsafe extern "C" fn(Display) -> i32),
        XRootWindow: load_sym!(XRootWindow, unsafe extern "C" fn(Display, i32) -> Window),
        XDefaultGC: load_sym!(
            XDefaultGC,
            unsafe extern "C" fn(Display, i32) -> *mut c_void
        ),
        XBlackPixel: load_sym!(XBlackPixel, unsafe extern "C" fn(Display, i32) -> u64),
        XWhitePixel: load_sym!(XWhitePixel, unsafe extern "C" fn(Display, i32) -> u64),
        XDisplayWidth: load_sym!(XDisplayWidth, unsafe extern "C" fn(Display, i32) -> i32),
        XDisplayHeight: load_sym!(XDisplayHeight, unsafe extern "C" fn(Display, i32) -> i32),
        XInternAtom: load_sym!(
            XInternAtom,
            unsafe extern "C" fn(Display, *const c_char, Bool) -> Atom
        ),
        XSetWMProtocols: load_sym!(
            XSetWMProtocols,
            unsafe extern "C" fn(Display, Window, *mut Atom, i32) -> i32
        ),
        XFlush: load_sym!(XFlush, unsafe extern "C" fn(Display) -> i32),
        XLookupKeysym: load_sym!(
            XLookupKeysym,
            unsafe extern "C" fn(*mut XEvent, i32) -> KeySym
        ),
        XGetWindowAttributes: load_sym!(
            XGetWindowAttributes,
            unsafe extern "C" fn(Display, Window, *mut XWindowAttributes) -> i32
        ),
        XDefaultColormap: load_sym!(
            XDefaultColormap,
            unsafe extern "C" fn(Display, i32) -> Colormap
        ),
    };

    X11 = Some(x11lib);
    X11.as_ref()
}

// ============================================================================
// X11Window
// ============================================================================

pub struct X11Window {
    display: *mut c_void, // X11 Display*
    window: u64,          // X11 Window (XID)
    screen: i32,
    width: u32,
    height: u32,
    pub resized: bool,
    should_close: bool,
    key_states: [bool; 256],
    mouse_x: f64,
    mouse_y: f64,
    mouse_buttons: [bool; 5],
    wm_delete_window: Atom,
    wm_protocols: Atom,
    visible: bool,
    pos_x: i32,
    pos_y: i32,
    pub events: crate::event::EventQueue,
}

impl X11Window {
    /// Create a window at the given position with the given size.
    /// `style` is a bitmask: 1=titled, 2=closable, 4=resizable, 8=miniaturizable, 32=frameless.
    pub unsafe fn create(title: &str, x: i32, y: i32, w: i32, h: i32, _style: i32) -> Option<Self> {
        let x11 = x11()?;

        let display = (x11.XOpenDisplay)(std::ptr::null());
        if display.is_null() {
            eprintln!("[rayzor-window] Cannot open X11 display. Is DISPLAY set?");
            return None;
        }

        let screen = (x11.XDefaultScreen)(display);
        let root = (x11.XRootWindow)(display, screen);
        let black = (x11.XBlackPixel)(display, screen);
        let white = (x11.XWhitePixel)(display, screen);

        let window = (x11.XCreateSimpleWindow)(
            display, root, x, y, w as u32, h as u32, 0,     // border_width
            black, // border color
            white, // background color
        );
        if window == 0 {
            (x11.XCloseDisplay)(display);
            return None;
        }

        // Set window title
        if let Ok(title_c) = CString::new(title) {
            (x11.XStoreName)(display, window, title_c.as_ptr());
        }

        // Select events
        (x11.XSelectInput)(display, window, EVENT_MASK);

        // Register WM_DELETE_WINDOW so the window manager sends ClientMessage on close
        let wm_protocols = (x11.XInternAtom)(
            display,
            b"WM_PROTOCOLS\0".as_ptr() as *const c_char,
            0, // False — create if needed
        );
        let mut wm_delete_window =
            (x11.XInternAtom)(display, b"WM_DELETE_WINDOW\0".as_ptr() as *const c_char, 0);
        (x11.XSetWMProtocols)(display, window, &mut wm_delete_window, 1);

        // Map (show) the window
        (x11.XMapWindow)(display, window);
        (x11.XFlush)(display);

        Some(X11Window {
            display,
            window,
            screen,
            width: w as u32,
            height: h as u32,
            resized: false,
            should_close: false,
            key_states: [false; 256],
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_buttons: [false; 5],
            wm_delete_window,
            wm_protocols,
            visible: true,
            pos_x: x,
            pos_y: y,
            events: crate::event::EventQueue::new(),
        })
    }

    /// Create a window centered on the screen.
    pub unsafe fn create_centered(title: &str, w: i32, h: i32) -> Option<Self> {
        let x11 = x11()?;

        // We need a temporary display connection to query screen dimensions
        let display = (x11.XOpenDisplay)(std::ptr::null());
        if display.is_null() {
            return None;
        }
        let screen = (x11.XDefaultScreen)(display);
        let screen_w = (x11.XDisplayWidth)(display, screen);
        let screen_h = (x11.XDisplayHeight)(display, screen);
        (x11.XCloseDisplay)(display);

        let x = (screen_w - w) / 2;
        let y = (screen_h - h) / 2;

        // Default style: titled + closable + resizable + miniaturizable
        Self::create(title, x, y, w, h, 1 | 2 | 4 | 8)
    }

    /// Drain all pending X11 events. Returns true if the window should remain open.
    pub unsafe fn poll_events(&mut self) -> bool {
        self.resized = false;
        self.events.clear();

        let x11 = match x11() {
            Some(x) => x,
            None => return false,
        };

        let mut event = XEvent::new();

        while (x11.XPending)(self.display) > 0 {
            (x11.XNextEvent)(self.display, &mut event);

            match event.event_type() {
                KEY_PRESS => {
                    let keysym = (x11.XLookupKeysym)(&mut event, 0);
                    let vk = x11_keysym_to_key(keysym);
                    if vk < 256 {
                        self.key_states[vk] = true;
                    }
                }
                KEY_RELEASE => {
                    let keysym = (x11.XLookupKeysym)(&mut event, 0);
                    let vk = x11_keysym_to_key(keysym);
                    if vk < 256 {
                        self.key_states[vk] = false;
                    }
                }
                BUTTON_PRESS => {
                    let btn = event.button();
                    match btn {
                        1 => self.mouse_buttons[0] = true, // left
                        2 => self.mouse_buttons[1] = true, // middle
                        3 => self.mouse_buttons[2] = true, // right
                        4 => self.mouse_buttons[3] = true, // scroll up
                        5 => self.mouse_buttons[4] = true, // scroll down
                        _ => {}
                    }
                }
                BUTTON_RELEASE => {
                    let btn = event.button();
                    match btn {
                        1 => self.mouse_buttons[0] = false,
                        2 => self.mouse_buttons[1] = false,
                        3 => self.mouse_buttons[2] = false,
                        4 => self.mouse_buttons[3] = false,
                        5 => self.mouse_buttons[4] = false,
                        _ => {}
                    }
                }
                MOTION_NOTIFY => {
                    self.mouse_x = event.motion_x() as f64;
                    self.mouse_y = event.motion_y() as f64;
                }
                CONFIGURE_NOTIFY => {
                    let new_w = event.configure_width() as u32;
                    let new_h = event.configure_height() as u32;
                    let new_x = event.configure_x();
                    let new_y = event.configure_y();

                    if new_w != self.width || new_h != self.height {
                        self.width = new_w;
                        self.height = new_h;
                        self.resized = true;
                    }
                    self.pos_x = new_x;
                    self.pos_y = new_y;
                }
                EXPOSE => {
                    // Redraw needed — for now just flush
                    (x11.XFlush)(self.display);
                }
                CLIENT_MESSAGE => {
                    // Check if this is WM_DELETE_WINDOW
                    if event.client_data_l0() == self.wm_delete_window {
                        self.should_close = true;
                    }
                }
                _ => {}
            }
        }

        !self.should_close
    }

    /// Check if a key is currently pressed. `key` uses cross-platform virtual key codes
    /// (matching the same scheme as CocoaWindow: ASCII values for letters, arrow keys, etc.).
    pub fn is_key_down(&self, key: i32) -> bool {
        (key >= 0 && key < 256) && self.key_states[key as usize]
    }

    /// Set the window title.
    pub unsafe fn set_title(&self, title: &str) {
        let x11 = match x11() {
            Some(x) => x,
            None => return,
        };
        if let Ok(title_c) = CString::new(title) {
            (x11.XStoreName)(self.display, self.window, title_c.as_ptr());
            (x11.XFlush)(self.display);
        }
    }

    /// Move the window to (x, y).
    pub unsafe fn set_position(&mut self, x: i32, y: i32) {
        let x11 = match x11() {
            Some(x) => x,
            None => return,
        };
        (x11.XMoveWindow)(self.display, self.window, x, y);
        (x11.XFlush)(self.display);
        self.pos_x = x;
        self.pos_y = y;
    }

    /// Resize the window to (w, h).
    pub unsafe fn set_size(&mut self, w: u32, h: u32) {
        let x11 = match x11() {
            Some(x) => x,
            None => return,
        };
        (x11.XResizeWindow)(self.display, self.window, w, h);
        (x11.XFlush)(self.display);
        self.width = w;
        self.height = h;
    }

    /// Show or hide the window.
    pub unsafe fn set_visible(&mut self, visible: bool) {
        let x11 = match x11() {
            Some(x) => x,
            None => return,
        };
        if visible {
            (x11.XMapWindow)(self.display, self.window);
        } else {
            (x11.XUnmapWindow)(self.display, self.window);
        }
        (x11.XFlush)(self.display);
        self.visible = visible;
    }

    /// Check if the window is currently visible (mapped).
    pub unsafe fn is_visible(&self) -> bool {
        let x11 = match x11() {
            Some(x) => x,
            None => return false,
        };
        let mut attrs = XWindowAttributes::zeroed();
        (x11.XGetWindowAttributes)(self.display, self.window, &mut attrs);
        // map_state: 0=Unmapped, 1=Unviewable, 2=IsViewable
        attrs.map_state == 2
    }

    /// Get the current window position.
    pub fn get_position(&self) -> (i32, i32) {
        (self.pos_x, self.pos_y)
    }

    /// Get the current mouse X position relative to the window.
    pub fn get_mouse_x(&self) -> f64 {
        self.mouse_x
    }

    /// Get the current mouse Y position relative to the window.
    pub fn get_mouse_y(&self) -> f64 {
        self.mouse_y
    }

    /// Check if a mouse button is pressed. 0=left, 1=middle, 2=right, 3=scroll up, 4=scroll down.
    pub fn is_mouse_down(&self, button: i32) -> bool {
        if button >= 0 && (button as usize) < self.mouse_buttons.len() {
            self.mouse_buttons[button as usize]
        } else {
            false
        }
    }

    // --- Stubs for missing features (TODO: implement with X11 calls) ---

    pub unsafe fn set_min_size(&mut self, _w: i32, _h: i32) {
        // TODO: XSetWMNormalHints with PMinSize
    }

    pub unsafe fn set_max_size(&mut self, _w: i32, _h: i32) {
        // TODO: XSetWMNormalHints with PMaxSize
    }

    pub unsafe fn set_fullscreen(&mut self, _fs: bool) {
        // TODO: _NET_WM_STATE_FULLSCREEN via XSendEvent
    }

    pub unsafe fn set_floating(&mut self, _on_top: bool) {
        // TODO: _NET_WM_STATE_ABOVE via XSendEvent
    }

    pub unsafe fn set_opacity(&mut self, _opacity: f64) {
        // TODO: _NET_WM_WINDOW_OPACITY via XChangeProperty
    }

    pub fn is_fullscreen(&self) -> bool {
        false
    }
    pub fn is_minimized(&self) -> bool {
        false
    }
    pub fn is_focused(&self) -> bool {
        true
    }

    /// Destroy the window and close the display connection.
    pub unsafe fn destroy(&mut self) {
        let x11 = match x11() {
            Some(x) => x,
            None => return,
        };
        if self.window != 0 {
            (x11.XDestroyWindow)(self.display, self.window);
            self.window = 0;
        }
        if !self.display.is_null() {
            (x11.XCloseDisplay)(self.display);
            self.display = std::ptr::null_mut();
        }
    }
}

// ============================================================================
// X11 keysym to cross-platform virtual key codes
// ============================================================================

/// Map X11 KeySym values to cross-platform key codes (matching CocoaWindow's scheme).
/// Uses standard virtual key codes: ASCII for letters/digits, well-known constants for specials.
fn x11_keysym_to_key(keysym: KeySym) -> usize {
    match keysym {
        // Escape, Space, Return, Tab, Backspace
        0xff1b => 27,  // XK_Escape
        0x0020 => 32,  // XK_space
        0xff0d => 13,  // XK_Return
        0xff09 => 9,   // XK_Tab
        0xff08 => 8,   // XK_BackSpace
        0xffff => 127, // XK_Delete

        // Arrow keys
        0xff51 => 37, // XK_Left
        0xff52 => 38, // XK_Up
        0xff53 => 39, // XK_Right
        0xff54 => 40, // XK_Down

        // Modifier keys
        0xffe1 => 160, // XK_Shift_L
        0xffe2 => 161, // XK_Shift_R
        0xffe3 => 162, // XK_Control_L
        0xffe4 => 163, // XK_Control_R
        0xffe9 => 164, // XK_Alt_L
        0xffea => 165, // XK_Alt_R
        0xffeb => 91,  // XK_Super_L  (Windows/Meta key)
        0xffec => 92,  // XK_Super_R

        // Function keys F1-F12
        0xffbe => 112, // XK_F1
        0xffbf => 113, // XK_F2
        0xffc0 => 114, // XK_F3
        0xffc1 => 115, // XK_F4
        0xffc2 => 116, // XK_F5
        0xffc3 => 117, // XK_F6
        0xffc4 => 118, // XK_F7
        0xffc5 => 119, // XK_F8
        0xffc6 => 120, // XK_F9
        0xffc7 => 121, // XK_F10
        0xffc8 => 122, // XK_F11
        0xffc9 => 123, // XK_F12

        // Navigation keys
        0xff50 => 36, // XK_Home
        0xff57 => 35, // XK_End
        0xff55 => 33, // XK_Page_Up
        0xff56 => 34, // XK_Page_Down
        0xff63 => 45, // XK_Insert

        // Caps Lock, Num Lock, Scroll Lock
        0xffe5 => 20,  // XK_Caps_Lock
        0xff7f => 144, // XK_Num_Lock
        0xff14 => 145, // XK_Scroll_Lock

        // Letters a-z / A-Z → uppercase ASCII (65-90)
        sym if (0x0061..=0x007a).contains(&sym) => (sym - 0x0061 + 65) as usize, // a-z
        sym if (0x0041..=0x005a).contains(&sym) => sym as usize,                 // A-Z

        // Digits 0-9
        sym if (0x0030..=0x0039).contains(&sym) => sym as usize,

        // Common punctuation mapped to ASCII
        sym if sym < 0x100 => sym as usize,

        _ => 0,
    }
}

// ============================================================================
// Tests (only compiled, not run — we're cross-compiling)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keysym_mapping() {
        assert_eq!(x11_keysym_to_key(0xff1b), 27); // Escape
        assert_eq!(x11_keysym_to_key(0xff0d), 13); // Return
        assert_eq!(x11_keysym_to_key(0x0061), 65); // 'a' -> 'A'
        assert_eq!(x11_keysym_to_key(0x0041), 65); // 'A' -> 65
        assert_eq!(x11_keysym_to_key(0x0030), 48); // '0'
        assert_eq!(x11_keysym_to_key(0xff51), 37); // Left arrow
        assert_eq!(x11_keysym_to_key(0xffbe), 112); // F1
    }

    #[test]
    fn test_xevent_size() {
        // XEvent must be 192 bytes on 64-bit systems
        assert_eq!(std::mem::size_of::<XEvent>(), 192);
    }
}
