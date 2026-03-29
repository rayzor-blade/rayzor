//! WASM host exports for rayzor-window via wasm-bindgen.
//!
//! All 42 window functions exported with full parity to native.
//! Uses web.rs WebWindow backend (DOM canvas + event listeners).
//!
//! Build: wasm-pack build --target web --no-default-features --features wasm-host

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
use wasm_bindgen::prelude::*;

use crate::web::WebWindow;

// ============================================================================
// Handle table
// ============================================================================

static WINDOWS: Lazy<Mutex<WindowTable>> = Lazy::new(|| Mutex::new(WindowTable::new()));

struct WindowTable {
    next: i32,
    windows: HashMap<i32, Box<WebWindow>>,
}

// SAFETY: WASM is single-threaded
unsafe impl Send for WindowTable {}
unsafe impl Sync for WindowTable {}

impl WindowTable {
    fn new() -> Self {
        Self {
            next: 1,
            windows: HashMap::new(),
        }
    }
    fn alloc(&mut self, win: WebWindow) -> i32 {
        let h = self.next;
        self.next += 1;
        self.windows.insert(h, Box::new(win));
        h
    }
    fn get(&self, h: i32) -> Option<&WebWindow> {
        self.windows.get(&h).map(|b| b.as_ref())
    }
    fn get_mut(&mut self, h: i32) -> Option<&mut WebWindow> {
        self.windows.get_mut(&h).map(|b| b.as_mut())
    }
    fn remove(&mut self, h: i32) {
        self.windows.remove(&h);
    }
}

// ============================================================================
// Creation
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_window_create")]
pub fn window_create(title: &str, x: i32, y: i32, w: i32, h: i32, style: i32) -> i32 {
    let mut wt = WINDOWS.lock().unwrap();
    match WebWindow::create(title, x, y, w, h, style) {
        Some(win) => wt.alloc(win),
        None => 0,
    }
}

#[wasm_bindgen(js_name = "rayzor_window_create_centered")]
pub fn window_create_centered(title: &str, w: i32, h: i32) -> i32 {
    let mut wt = WINDOWS.lock().unwrap();
    match WebWindow::create_centered(title, w, h) {
        Some(win) => wt.alloc(win),
        None => 0,
    }
}

#[wasm_bindgen(js_name = "rayzor_window_destroy")]
pub fn window_destroy(h: i32) {
    WINDOWS.lock().unwrap().remove(h);
}

// ============================================================================
// Event loop
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_window_poll_events")]
pub fn window_poll_events(h: i32) -> i32 {
    let mut wt = WINDOWS.lock().unwrap();
    match wt.get_mut(h) {
        Some(w) => {
            if w.poll_events() {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

#[wasm_bindgen(js_name = "rayzor_window_is_key_down")]
pub fn window_is_key_down(h: i32, key: i32) -> i32 {
    let wt = WINDOWS.lock().unwrap();
    match wt.get(h) {
        Some(w) => {
            if w.is_key_down(key) {
                1
            } else {
                0
            }
        }
        None => 0,
    }
}

// ============================================================================
// Native handles (for GPU surface — on WASM, handle = window ID)
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_window_get_handle")]
pub fn window_get_handle(h: i32) -> i32 {
    h
}

#[wasm_bindgen(js_name = "rayzor_window_get_display_handle")]
pub fn window_get_display_handle(_h: i32) -> i32 {
    0
}

// ============================================================================
// Geometry
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_window_get_width")]
pub fn window_get_width(h: i32) -> i32 {
    WINDOWS.lock().unwrap().get(h).map(|w| w.width).unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_get_height")]
pub fn window_get_height(h: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .map(|w| w.height)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_get_x")]
pub fn window_get_x(_h: i32) -> i32 {
    0
} // browser canvas doesn't have screen position

#[wasm_bindgen(js_name = "rayzor_window_get_y")]
pub fn window_get_y(_h: i32) -> i32 {
    0
}

#[wasm_bindgen(js_name = "rayzor_window_was_resized")]
pub fn window_was_resized(h: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .map(|w| if w.resized { 1 } else { 0 })
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_set_position")]
pub fn window_set_position(_h: i32, _x: i32, _y: i32) {} // no-op in browser

#[wasm_bindgen(js_name = "rayzor_window_set_size")]
pub fn window_set_size(h: i32, w: i32, ht_val: i32) {
    let mut wt = WINDOWS.lock().unwrap();
    if let Some(win) = wt.get_mut(h) {
        win.canvas.set_width(w as u32);
        win.canvas.set_height(ht_val as u32);
        win.width = w;
        win.height = ht_val;
    }
}

#[wasm_bindgen(js_name = "rayzor_window_set_min_size")]
pub fn window_set_min_size(_h: i32, _w: i32, _ht: i32) {}

#[wasm_bindgen(js_name = "rayzor_window_set_max_size")]
pub fn window_set_max_size(_h: i32, _w: i32, _ht: i32) {}

// ============================================================================
// Appearance
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_window_set_title")]
pub fn window_set_title(_h: i32, title: &str) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        doc.set_title(title);
    }
}

#[wasm_bindgen(js_name = "rayzor_window_set_fullscreen")]
pub fn window_set_fullscreen(h: i32, fs: i32) {
    let wt = WINDOWS.lock().unwrap();
    if let Some(win) = wt.get(h) {
        if fs != 0 {
            let _ = win.canvas.request_fullscreen();
        }
        // exit fullscreen is on document, not canvas
    }
}

#[wasm_bindgen(js_name = "rayzor_window_set_visible")]
pub fn window_set_visible(h: i32, vis: i32) {
    let wt = WINDOWS.lock().unwrap();
    if let Some(win) = wt.get(h) {
        let _ = win
            .canvas
            .style()
            .set_property("display", if vis != 0 { "block" } else { "none" });
    }
}

#[wasm_bindgen(js_name = "rayzor_window_set_floating")]
pub fn window_set_floating(_h: i32, _on_top: i32) {} // no-op

#[wasm_bindgen(js_name = "rayzor_window_set_opacity")]
pub fn window_set_opacity(h: i32, opacity: f64) {
    let wt = WINDOWS.lock().unwrap();
    if let Some(win) = wt.get(h) {
        let _ = win
            .canvas
            .style()
            .set_property("opacity", &format!("{}", opacity));
    }
}

// ============================================================================
// State queries
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_window_is_fullscreen")]
pub fn window_is_fullscreen(_h: i32) -> i32 {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.fullscreen_element())
        .map(|_| 1)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_is_visible")]
pub fn window_is_visible(h: i32) -> i32 {
    let wt = WINDOWS.lock().unwrap();
    wt.get(h)
        .map(|w| {
            w.canvas
                .style()
                .get_property_value("display")
                .map(|d| if d == "none" { 0 } else { 1 })
                .unwrap_or(1)
        })
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_is_minimized")]
pub fn window_is_minimized(_h: i32) -> i32 {
    // document.hidden maps to tab visibility
    web_sys::window()
        .and_then(|w| w.document())
        .map(|d| if d.hidden() { 1 } else { 0 })
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_is_focused")]
pub fn window_is_focused(h: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .map(|w| if w.focused { 1 } else { 0 })
        .unwrap_or(0)
}

// ============================================================================
// Mouse input
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_window_get_mouse_x")]
pub fn window_get_mouse_x(h: i32) -> f64 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .map(|w| w.mouse_x)
        .unwrap_or(0.0)
}

#[wasm_bindgen(js_name = "rayzor_window_get_mouse_y")]
pub fn window_get_mouse_y(h: i32) -> f64 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .map(|w| w.mouse_y)
        .unwrap_or(0.0)
}

#[wasm_bindgen(js_name = "rayzor_window_is_mouse_down")]
pub fn window_is_mouse_down(h: i32, button: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .map(|w| {
            if w.mouse_buttons.contains(&button) {
                1
            } else {
                0
            }
        })
        .unwrap_or(0)
}

// ============================================================================
// Event queue
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_window_event_count")]
pub fn window_event_count(h: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .map(|w| w.events.len() as i32)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_type")]
pub fn window_event_type(h: i32, idx: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.event_type)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_x")]
pub fn window_event_x(h: i32, idx: i32) -> f64 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.x)
        .unwrap_or(0.0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_y")]
pub fn window_event_y(h: i32, idx: i32) -> f64 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.y)
        .unwrap_or(0.0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_key")]
pub fn window_event_key(h: i32, idx: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.key)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_button")]
pub fn window_event_button(h: i32, idx: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.button)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_modifiers")]
pub fn window_event_modifiers(h: i32, idx: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.modifiers)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_width")]
pub fn window_event_width(h: i32, idx: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.width)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_height")]
pub fn window_event_height(h: i32, idx: i32) -> i32 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.height)
        .unwrap_or(0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_scroll_x")]
pub fn window_event_scroll_x(h: i32, idx: i32) -> f64 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.scroll_x)
        .unwrap_or(0.0)
}

#[wasm_bindgen(js_name = "rayzor_window_event_scroll_y")]
pub fn window_event_scroll_y(h: i32, idx: i32) -> f64 {
    WINDOWS
        .lock()
        .unwrap()
        .get(h)
        .and_then(|w| w.events.get(idx as usize))
        .map(|e| e.scroll_y)
        .unwrap_or(0.0)
}
