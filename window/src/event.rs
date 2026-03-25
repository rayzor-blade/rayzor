//! Window event types shared across all platform backends.

/// Event type IDs (must match rayzor.window.EventType in Haxe)
pub mod event_type {
    pub const NONE: i32 = 0;
    pub const MOUSE_MOVE: i32 = 1;
    pub const MOUSE_DOWN: i32 = 2;
    pub const MOUSE_UP: i32 = 3;
    pub const MOUSE_WHEEL: i32 = 4;
    pub const KEY_DOWN: i32 = 5;
    pub const KEY_UP: i32 = 6;
    pub const RESIZE: i32 = 7;
    pub const MOVE: i32 = 8;
    pub const FOCUS: i32 = 9;
    pub const BLUR: i32 = 10;
    pub const CLOSE: i32 = 11;
    pub const MOUSE_ENTER: i32 = 12;
    pub const MOUSE_LEAVE: i32 = 13;
}

/// A window event with all possible fields.
/// Unused fields are zeroed. Lightweight — no heap allocation.
#[derive(Clone, Copy)]
pub struct WindowEvent {
    pub event_type: i32,
    pub x: f64,
    pub y: f64,
    pub button: i32,
    pub key: i32,
    pub modifiers: i32, // bitmask: 1=shift, 2=ctrl, 4=alt, 8=meta/cmd
    pub width: i32,
    pub height: i32,
    pub scroll_x: f64,
    pub scroll_y: f64,
}

impl WindowEvent {
    pub fn mouse_move(x: f64, y: f64) -> Self {
        Self { event_type: event_type::MOUSE_MOVE, x, y, ..Self::ZERO }
    }
    pub fn mouse_down(button: i32, x: f64, y: f64) -> Self {
        Self { event_type: event_type::MOUSE_DOWN, button, x, y, ..Self::ZERO }
    }
    pub fn mouse_up(button: i32, x: f64, y: f64) -> Self {
        Self { event_type: event_type::MOUSE_UP, button, x, y, ..Self::ZERO }
    }
    pub fn mouse_wheel(dx: f64, dy: f64) -> Self {
        Self { event_type: event_type::MOUSE_WHEEL, scroll_x: dx, scroll_y: dy, ..Self::ZERO }
    }
    pub fn key_down(key: i32, modifiers: i32) -> Self {
        Self { event_type: event_type::KEY_DOWN, key, modifiers, ..Self::ZERO }
    }
    pub fn key_up(key: i32, modifiers: i32) -> Self {
        Self { event_type: event_type::KEY_UP, key, modifiers, ..Self::ZERO }
    }
    pub fn resize(w: i32, h: i32) -> Self {
        Self { event_type: event_type::RESIZE, width: w, height: h, ..Self::ZERO }
    }
    pub fn window_move(x: i32, y: i32) -> Self {
        Self { event_type: event_type::MOVE, x: x as f64, y: y as f64, ..Self::ZERO }
    }
    pub fn focus() -> Self {
        Self { event_type: event_type::FOCUS, ..Self::ZERO }
    }
    pub fn blur() -> Self {
        Self { event_type: event_type::BLUR, ..Self::ZERO }
    }
    pub fn close() -> Self {
        Self { event_type: event_type::CLOSE, ..Self::ZERO }
    }

    const ZERO: Self = Self {
        event_type: 0, x: 0.0, y: 0.0, button: 0, key: 0,
        modifiers: 0, width: 0, height: 0, scroll_x: 0.0, scroll_y: 0.0,
    };
}

/// Event queue — accumulated during poll_events(), read by Haxe.
pub struct EventQueue {
    events: Vec<WindowEvent>,
}

impl EventQueue {
    pub fn new() -> Self {
        Self { events: Vec::with_capacity(32) }
    }

    pub fn push(&mut self, event: WindowEvent) {
        self.events.push(event);
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn get(&self, index: usize) -> Option<&WindowEvent> {
        self.events.get(index)
    }
}
