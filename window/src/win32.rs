//! Windows window implementation via raw Win32 FFI.
//!
//! Uses user32.dll and kernel32.dll directly — no third-party windowing crates.
//! All string parameters use UTF-16 (wide) encoding via `to_wide()`.

#![cfg(target_os = "windows")]

use std::ffi::c_void;

// ============================================================================
// Win32 type aliases and constants
// ============================================================================

const WS_OVERLAPPEDWINDOW: u32 = 0x00CF0000;
const WS_VISIBLE: u32 = 0x10000000;
const WS_EX_LAYERED: u32 = 0x00080000;

const WM_CLOSE: u32 = 0x0010;
const WM_DESTROY: u32 = 0x0002;
const WM_SIZE: u32 = 0x0005;
const WM_KEYDOWN: u32 = 0x0100;
const WM_KEYUP: u32 = 0x0101;
const WM_SYSKEYDOWN: u32 = 0x0104;
const WM_SYSKEYUP: u32 = 0x0105;
const WM_LBUTTONDOWN: u32 = 0x0201;
const WM_LBUTTONUP: u32 = 0x0202;
const WM_RBUTTONDOWN: u32 = 0x0204;
const WM_RBUTTONUP: u32 = 0x0205;
const WM_MBUTTONDOWN: u32 = 0x0207;
const WM_MBUTTONUP: u32 = 0x0208;
const WM_XBUTTONDOWN: u32 = 0x020B;
const WM_XBUTTONUP: u32 = 0x020C;
const WM_MOUSEMOVE: u32 = 0x0200;

const PM_REMOVE: u32 = 0x0001;
const SW_SHOW: i32 = 5;
const SW_HIDE: i32 = 0;
const SW_MAXIMIZE: i32 = 3;
const SW_RESTORE: i32 = 9;

const SM_CXSCREEN: i32 = 0;
const SM_CYSCREEN: i32 = 1;

const GWLP_USERDATA: i32 = -21;
const GWL_STYLE: i32 = -16;
const GWL_EXSTYLE: i32 = -20;

const CS_HREDRAW: u32 = 0x0002;
const CS_VREDRAW: u32 = 0x0001;

const CW_USEDEFAULT: i32 = 0x80000000u32 as i32;

const LWA_ALPHA: u32 = 0x00000002;

const SWP_NOMOVE: u32 = 0x0002;
const SWP_NOSIZE: u32 = 0x0001;
const SWP_NOZORDER: u32 = 0x0004;
const SWP_FRAMECHANGED: u32 = 0x0020;

const HWND_TOP: *mut c_void = 0 as *mut c_void;

const IDC_ARROW: *const u16 = 32512 as *const u16;

// ============================================================================
// Win32 structs
// ============================================================================

#[repr(C)]
struct WNDCLASSEXW {
    cb_size: u32,
    style: u32,
    lpfn_wnd_proc: unsafe extern "system" fn(*mut c_void, u32, usize, isize) -> isize,
    cb_cls_extra: i32,
    cb_wnd_extra: i32,
    h_instance: *mut c_void,
    h_icon: *mut c_void,
    h_cursor: *mut c_void,
    hbr_background: *mut c_void,
    lpsz_menu_name: *const u16,
    lpsz_class_name: *const u16,
    h_icon_sm: *mut c_void,
}

#[repr(C)]
struct MSG {
    hwnd: *mut c_void,
    message: u32,
    w_param: usize,
    l_param: isize,
    time: u32,
    pt: POINT,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct POINT {
    x: i32,
    y: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RECT {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

// ============================================================================
// Win32 API extern declarations
// ============================================================================

#[link(name = "user32")]
extern "system" {
    fn RegisterClassExW(lpWndClass: *const WNDCLASSEXW) -> u16;
    fn CreateWindowExW(
        dwExStyle: u32,
        lpClassName: *const u16,
        lpWindowName: *const u16,
        dwStyle: u32,
        X: i32,
        Y: i32,
        nWidth: i32,
        nHeight: i32,
        hWndParent: *mut c_void,
        hMenu: *mut c_void,
        hInstance: *mut c_void,
        lpParam: *mut c_void,
    ) -> *mut c_void;
    fn ShowWindow(hWnd: *mut c_void, nCmdShow: i32) -> i32;
    fn UpdateWindow(hWnd: *mut c_void) -> i32;
    fn PeekMessageW(
        lpMsg: *mut MSG,
        hWnd: *mut c_void,
        wMsgFilterMin: u32,
        wMsgFilterMax: u32,
        wRemoveMsg: u32,
    ) -> i32;
    fn TranslateMessage(lpMsg: *const MSG) -> i32;
    fn DispatchMessageW(lpMsg: *const MSG) -> isize;
    fn DestroyWindow(hWnd: *mut c_void) -> i32;
    fn DefWindowProcW(hWnd: *mut c_void, Msg: u32, wParam: usize, lParam: isize) -> isize;
    fn PostQuitMessage(nExitCode: i32);
    fn SetWindowTextW(hWnd: *mut c_void, lpString: *const u16) -> i32;
    fn MoveWindow(
        hWnd: *mut c_void,
        X: i32,
        Y: i32,
        nWidth: i32,
        nHeight: i32,
        bRepaint: i32,
    ) -> i32;
    fn SetWindowPos(
        hWnd: *mut c_void,
        hWndInsertAfter: *mut c_void,
        X: i32,
        Y: i32,
        cx: i32,
        cy: i32,
        uFlags: u32,
    ) -> i32;
    fn GetSystemMetrics(nIndex: i32) -> i32;
    fn GetClientRect(hWnd: *mut c_void, lpRect: *mut RECT) -> i32;
    fn GetWindowRect(hWnd: *mut c_void, lpRect: *mut RECT) -> i32;
    fn GetCursorPos(lpPoint: *mut POINT) -> i32;
    fn ScreenToClient(hWnd: *mut c_void, lpPoint: *mut POINT) -> i32;
    fn IsWindowVisible(hWnd: *mut c_void) -> i32;
    fn IsIconic(hWnd: *mut c_void) -> i32;
    fn GetForegroundWindow() -> *mut c_void;
    fn SetLayeredWindowAttributes(
        hWnd: *mut c_void,
        crKey: u32,
        bAlpha: u8,
        dwFlags: u32,
    ) -> i32;
    fn SetWindowLongW(hWnd: *mut c_void, nIndex: i32, dwNewLong: i32) -> i32;
    fn GetWindowLongW(hWnd: *mut c_void, nIndex: i32) -> i32;
    fn LoadCursorW(hInstance: *mut c_void, lpCursorName: *const u16) -> *mut c_void;
}

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleW(lpModuleName: *const u16) -> *mut c_void;
}

// On 64-bit Windows, GWLP_USERDATA requires the Ptr-width variants.
// SetWindowLongPtrW/GetWindowLongPtrW are macros in C headers that resolve
// to SetWindowLongW/GetWindowLongW on 32-bit. On 64-bit they are distinct
// functions. We declare them separately.
#[cfg(target_pointer_width = "64")]
#[link(name = "user32")]
extern "system" {
    fn SetWindowLongPtrW(hWnd: *mut c_void, nIndex: i32, dwNewLong: isize) -> isize;
    fn GetWindowLongPtrW(hWnd: *mut c_void, nIndex: i32) -> isize;
}

#[cfg(target_pointer_width = "32")]
unsafe fn SetWindowLongPtrW(hWnd: *mut c_void, nIndex: i32, dwNewLong: isize) -> isize {
    SetWindowLongW(hWnd, nIndex, dwNewLong as i32) as isize
}

#[cfg(target_pointer_width = "32")]
unsafe fn GetWindowLongPtrW(hWnd: *mut c_void, nIndex: i32) -> isize {
    GetWindowLongW(hWnd, nIndex) as isize
}

// ============================================================================
// UTF-16 helper
// ============================================================================

fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0u16)).collect()
}

// ============================================================================
// Window procedure
// ============================================================================

unsafe extern "system" fn window_proc(
    hwnd: *mut c_void,
    msg: u32,
    w_param: usize,
    l_param: isize,
) -> isize {
    let user_data = GetWindowLongPtrW(hwnd, GWLP_USERDATA);
    if user_data == 0 {
        return DefWindowProcW(hwnd, msg, w_param, l_param);
    }

    let win = &mut *(user_data as *mut Win32Window);

    match msg {
        WM_CLOSE => {
            win.should_close = true;
            // Return 0 to prevent DefWindowProcW from calling DestroyWindow.
            // The application controls the window lifetime via destroy().
            return 0;
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            return 0;
        }
        WM_SIZE => {
            let width = (l_param as u32) & 0xFFFF;
            let height = ((l_param as u32) >> 16) & 0xFFFF;
            if width != win.width || height != win.height {
                win.width = width;
                win.height = height;
                win.resized = true;
            }
            return 0;
        }
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            let vk = w_param & 0xFF;
            if vk < 256 {
                win.key_states[vk] = true;
            }
            return 0;
        }
        WM_KEYUP | WM_SYSKEYUP => {
            let vk = w_param & 0xFF;
            if vk < 256 {
                win.key_states[vk] = false;
            }
            return 0;
        }
        WM_LBUTTONDOWN => {
            win.mouse_buttons[0] = true;
            return 0;
        }
        WM_LBUTTONUP => {
            win.mouse_buttons[0] = false;
            return 0;
        }
        WM_RBUTTONDOWN => {
            win.mouse_buttons[1] = true;
            return 0;
        }
        WM_RBUTTONUP => {
            win.mouse_buttons[1] = false;
            return 0;
        }
        WM_MBUTTONDOWN => {
            win.mouse_buttons[2] = true;
            return 0;
        }
        WM_MBUTTONUP => {
            win.mouse_buttons[2] = false;
            return 0;
        }
        WM_XBUTTONDOWN => {
            let button = ((w_param >> 16) & 0xFFFF) as u32;
            if button == 1 {
                win.mouse_buttons[3] = true;
            } else if button == 2 {
                win.mouse_buttons[4] = true;
            }
            return 0;
        }
        WM_XBUTTONUP => {
            let button = ((w_param >> 16) & 0xFFFF) as u32;
            if button == 1 {
                win.mouse_buttons[3] = false;
            } else if button == 2 {
                win.mouse_buttons[4] = false;
            }
            return 0;
        }
        WM_MOUSEMOVE => {
            let x = (l_param & 0xFFFF) as i16 as f64;
            let y = ((l_param >> 16) & 0xFFFF) as i16 as f64;
            win.mouse_x = x;
            win.mouse_y = y;
            return 0;
        }
        _ => {}
    }

    DefWindowProcW(hwnd, msg, w_param, l_param)
}

// ============================================================================
// Win32Window
// ============================================================================

pub struct Win32Window {
    pub(crate) hwnd: *mut c_void,
    hinstance: *mut c_void,
    pub width: u32,
    pub height: u32,
    pub resized: bool,
    pub should_close: bool,
    key_states: [bool; 256],
    mouse_x: f64,
    mouse_y: f64,
    mouse_buttons: [bool; 5],
    // Saved state for fullscreen toggle
    pre_fullscreen_style: u32,
    pre_fullscreen_rect: RECT,
    is_fullscreen: bool,
    pub events: crate::event::EventQueue,
}

static CLASS_REGISTERED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

const RAYZOR_CLASS_NAME: &str = "RayzorWindowClass";

impl Win32Window {
    /// Register the window class (once per process).
    unsafe fn ensure_class_registered(hinstance: *mut c_void) {
        if CLASS_REGISTERED.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }

        let class_name = to_wide(RAYZOR_CLASS_NAME);
        let cursor = LoadCursorW(std::ptr::null_mut(), IDC_ARROW);

        let wc = WNDCLASSEXW {
            cb_size: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfn_wnd_proc: window_proc,
            cb_cls_extra: 0,
            cb_wnd_extra: 0,
            h_instance: hinstance,
            h_icon: std::ptr::null_mut(),
            h_cursor: cursor,
            hbr_background: std::ptr::null_mut(),
            lpsz_menu_name: std::ptr::null(),
            lpsz_class_name: class_name.as_ptr(),
            h_icon_sm: std::ptr::null_mut(),
        };

        RegisterClassExW(&wc);
        CLASS_REGISTERED.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Create a window at an explicit position.
    pub unsafe fn create(
        title: &str,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        _style: i32,
    ) -> Option<Self> {
        let hinstance = GetModuleHandleW(std::ptr::null());
        if hinstance.is_null() {
            return None;
        }

        Self::ensure_class_registered(hinstance);

        let class_name = to_wide(RAYZOR_CLASS_NAME);
        let window_title = to_wide(title);

        let dw_style = WS_OVERLAPPEDWINDOW | WS_VISIBLE;

        let hwnd = CreateWindowExW(
            0,
            class_name.as_ptr(),
            window_title.as_ptr(),
            dw_style,
            x,
            y,
            w,
            h,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            hinstance,
            std::ptr::null_mut(),
        );

        if hwnd.is_null() {
            return None;
        }

        let mut win = Win32Window {
            hwnd,
            hinstance,
            width: w as u32,
            height: h as u32,
            resized: false,
            should_close: false,
            key_states: [false; 256],
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_buttons: [false; 5],
            pre_fullscreen_style: 0,
            pre_fullscreen_rect: RECT {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            },
            is_fullscreen: false,
            events: crate::event::EventQueue::new(),
        };

        // Store pointer to self as GWLP_USERDATA so the window proc can access it.
        // We use a raw pointer — the caller must ensure the Win32Window outlives the HWND.
        let win_ptr = &mut win as *mut Win32Window;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, win_ptr as isize);

        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);

        // Read actual client size after creation (may differ from requested due to borders).
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        if GetClientRect(hwnd, &mut rect) != 0 {
            win.width = (rect.right - rect.left) as u32;
            win.height = (rect.bottom - rect.top) as u32;
        }

        Some(win)
    }

    /// Create a window centered on the primary monitor.
    pub unsafe fn create_centered(title: &str, w: i32, h: i32) -> Option<Self> {
        let screen_w = GetSystemMetrics(SM_CXSCREEN);
        let screen_h = GetSystemMetrics(SM_CYSCREEN);
        let x = (screen_w - w) / 2;
        let y = (screen_h - h) / 2;
        Self::create(title, x, y, w, h, 0)
    }

    /// Pump the message queue. Returns true if the window is still alive
    /// (not closed, not destroyed).
    pub unsafe fn poll_events(&mut self) -> bool {
        self.resized = false;
        self.events.clear();

        // Re-register our pointer in case the struct was moved (e.g., Box realloc).
        SetWindowLongPtrW(self.hwnd, GWLP_USERDATA, self as *mut Win32Window as isize);

        let mut msg: MSG = std::mem::zeroed();
        while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
            // WM_QUIT means the message loop should end.
            if msg.message == 0x0012 {
                // WM_QUIT
                self.should_close = true;
                return false;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        !self.should_close
    }

    /// Check if a virtual key code is currently pressed.
    /// Win32 virtual key codes map mostly 1:1 to our key indices:
    ///   VK_ESCAPE=0x1B(27), VK_SPACE=0x20(32), VK_RETURN=0x0D(13), etc.
    ///   A-Z = 0x41-0x5A (65-90), 0-9 = 0x30-0x39 (48-57)
    ///   VK_LEFT=0x25(37), VK_UP=0x26(38), VK_RIGHT=0x27(39), VK_DOWN=0x28(40)
    ///   VK_F1-F12 = 0x70-0x7B (112-123)
    pub fn is_key_down(&self, key: i32) -> bool {
        key >= 0 && key < 256 && self.key_states[key as usize]
    }

    /// Set the window title.
    pub unsafe fn set_title(&self, title: &str) {
        let wide = to_wide(title);
        SetWindowTextW(self.hwnd, wide.as_ptr());
    }

    /// Move the window to a screen position.
    pub unsafe fn set_position(&self, x: i32, y: i32) {
        SetWindowPos(
            self.hwnd,
            HWND_TOP,
            x,
            y,
            0,
            0,
            SWP_NOSIZE | SWP_NOZORDER,
        );
    }

    /// Resize the window (outer dimensions).
    pub unsafe fn set_size(&self, w: i32, h: i32) {
        SetWindowPos(
            self.hwnd,
            HWND_TOP,
            0,
            0,
            w,
            h,
            SWP_NOMOVE | SWP_NOZORDER,
        );
    }

    /// Show or hide the window.
    pub unsafe fn set_visible(&self, visible: bool) {
        ShowWindow(self.hwnd, if visible { SW_SHOW } else { SW_HIDE });
    }

    /// Check if the window is currently visible.
    pub unsafe fn is_visible(&self) -> bool {
        IsWindowVisible(self.hwnd) != 0
    }

    /// Check if the window is minimized (iconic).
    pub unsafe fn is_minimized(&self) -> bool {
        IsIconic(self.hwnd) != 0
    }

    /// Check if this window is the foreground (focused) window.
    pub unsafe fn is_focused(&self) -> bool {
        GetForegroundWindow() == self.hwnd
    }

    /// Toggle borderless fullscreen.
    ///
    /// Entering fullscreen: saves current style and rect, removes
    /// WS_OVERLAPPEDWINDOW, maximizes to cover the entire screen.
    /// Exiting fullscreen: restores the saved style and rect.
    pub unsafe fn set_fullscreen(&mut self, fullscreen: bool) {
        if fullscreen == self.is_fullscreen {
            return;
        }

        if fullscreen {
            // Save current state
            self.pre_fullscreen_style = GetWindowLongW(self.hwnd, GWL_STYLE) as u32;
            GetWindowRect(self.hwnd, &mut self.pre_fullscreen_rect);

            // Remove overlapped window chrome, maximize to screen
            let new_style = self.pre_fullscreen_style & !WS_OVERLAPPEDWINDOW;
            SetWindowLongW(self.hwnd, GWL_STYLE, new_style as i32);

            let screen_w = GetSystemMetrics(SM_CXSCREEN);
            let screen_h = GetSystemMetrics(SM_CYSCREEN);
            SetWindowPos(
                self.hwnd,
                HWND_TOP,
                0,
                0,
                screen_w,
                screen_h,
                SWP_FRAMECHANGED,
            );
            ShowWindow(self.hwnd, SW_MAXIMIZE);
        } else {
            // Restore saved style and position
            SetWindowLongW(self.hwnd, GWL_STYLE, self.pre_fullscreen_style as i32);
            let r = &self.pre_fullscreen_rect;
            SetWindowPos(
                self.hwnd,
                HWND_TOP,
                r.left,
                r.top,
                r.right - r.left,
                r.bottom - r.top,
                SWP_FRAMECHANGED,
            );
            ShowWindow(self.hwnd, SW_RESTORE);
        }

        self.is_fullscreen = fullscreen;
    }

    /// Set window opacity (0.0 = transparent, 1.0 = opaque).
    ///
    /// Adds WS_EX_LAYERED extended style and uses SetLayeredWindowAttributes
    /// with LWA_ALPHA. Setting opacity to 1.0 removes the layered style
    /// for best performance.
    pub unsafe fn set_opacity(&self, opacity: f64) {
        let alpha = (opacity.clamp(0.0, 1.0) * 255.0) as u8;

        if alpha == 255 {
            // Fully opaque — remove layered flag for performance
            let ex_style = GetWindowLongW(self.hwnd, GWL_EXSTYLE) as u32;
            SetWindowLongW(self.hwnd, GWL_EXSTYLE, (ex_style & !WS_EX_LAYERED) as i32);
        } else {
            // Add layered flag and set alpha
            let ex_style = GetWindowLongW(self.hwnd, GWL_EXSTYLE) as u32;
            SetWindowLongW(self.hwnd, GWL_EXSTYLE, (ex_style | WS_EX_LAYERED) as i32);
            SetLayeredWindowAttributes(self.hwnd, 0, alpha, LWA_ALPHA);
        }
    }

    /// Get the current mouse X position relative to the client area.
    pub unsafe fn get_mouse_x(&self) -> f64 {
        self.mouse_x
    }

    /// Get the current mouse Y position relative to the client area.
    pub unsafe fn get_mouse_y(&self) -> f64 {
        self.mouse_y
    }

    /// Check if a mouse button is currently pressed.
    /// 0 = left, 1 = right, 2 = middle, 3 = X1, 4 = X2.
    pub fn is_mouse_down(&self, button: i32) -> bool {
        button >= 0 && button < 5 && self.mouse_buttons[button as usize]
    }

    pub unsafe fn get_position(&self) -> (i32, i32) {
        let mut rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        GetWindowRect(self.hwnd, &mut rect);
        (rect.left, rect.top)
    }

    pub unsafe fn set_min_size(&mut self, _w: i32, _h: i32) {
        // TODO: handle WM_GETMINMAXINFO
    }

    pub unsafe fn set_max_size(&mut self, _w: i32, _h: i32) {
        // TODO: handle WM_GETMINMAXINFO
    }

    pub unsafe fn set_floating(&self, on_top: bool) {
        let insert_after = if on_top { -1isize as *mut c_void } else { -2isize as *mut c_void }; // HWND_TOPMOST / HWND_NOTOPMOST
        SetWindowPos(self.hwnd, insert_after, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040); // SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
    }

    pub fn is_fullscreen(&self) -> bool {
        self.is_fullscreen
    }

    /// Destroy the window and clean up.
    pub unsafe fn destroy(&mut self) {
        if !self.hwnd.is_null() {
            // Clear user data to prevent the window proc from using a dangling pointer.
            SetWindowLongPtrW(self.hwnd, GWLP_USERDATA, 0);
            DestroyWindow(self.hwnd);
            self.hwnd = std::ptr::null_mut();
        }
    }
}
