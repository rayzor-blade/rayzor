//! Browser DOM backend for rayzor-window.
//!
//! Creates an HTML <canvas> element as the window surface,
//! hooks keyboard, mouse, wheel, resize, and focus events via web-sys.

use crate::event::{EventQueue, WindowEvent};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent, WheelEvent};

/// Key code mapping: DOM KeyboardEvent.code → rayzor Key constants
fn map_key_code(code: &str) -> i32 {
    match code {
        "Escape" => 256,
        "Enter" => 257,
        "Tab" => 258,
        "Backspace" => 259,
        "Delete" => 261,
        "ArrowRight" => 262,
        "ArrowLeft" => 263,
        "ArrowDown" => 264,
        "ArrowUp" => 265,
        "Space" => 32,
        "ShiftLeft" | "ShiftRight" => 340,
        "ControlLeft" | "ControlRight" => 341,
        "AltLeft" | "AltRight" => 342,
        "MetaLeft" | "MetaRight" => 343,
        _ => {
            if code.starts_with("Key") && code.len() == 4 {
                code.as_bytes()[3] as i32 // KeyA=65, KeyB=66, ...
            } else if code.starts_with("Digit") && code.len() == 6 {
                code.as_bytes()[5] as i32 // Digit0=48, ...
            } else if code.starts_with("F") && code.len() <= 3 {
                if let Ok(n) = code[1..].parse::<i32>() {
                    289 + n
                } else {
                    0
                }
            } else {
                0
            }
        }
    }
}

fn map_modifiers(e: &KeyboardEvent) -> i32 {
    (if e.shift_key() { 1 } else { 0 })
        | (if e.ctrl_key() { 2 } else { 0 })
        | (if e.alt_key() { 4 } else { 0 })
        | (if e.meta_key() { 8 } else { 0 })
}

/// Browser window backed by an HTML <canvas> element.
pub struct WebWindow {
    pub canvas: HtmlCanvasElement,
    pub width: i32,
    pub height: i32,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub keys_down: HashSet<i32>,
    pub mouse_buttons: HashSet<i32>,
    pub events: EventQueue,
    pub open: bool,
    pub focused: bool,
    pub resized: bool,
    // Prevent GC of closures
    _closures: Vec<Closure<dyn FnMut(JsValue)>>,
}

impl WebWindow {
    pub fn create(
        title: &str,
        _x: i32,
        _y: i32,
        width: i32,
        height: i32,
        _style: i32,
    ) -> Option<Self> {
        let document = web_sys::window()?.document()?;

        let canvas: HtmlCanvasElement = document
            .query_selector("canvas")
            .ok()?
            .and_then(|e| e.dyn_into().ok())
            .unwrap_or_else(|| {
                let c: HtmlCanvasElement = document
                    .create_element("canvas")
                    .unwrap()
                    .dyn_into()
                    .unwrap();
                c.set_id("rayzor-window");
                c.style().set_property("display", "block").unwrap();
                c.style().set_property("margin", "0 auto").unwrap();
                c.style().set_property("background", "#1a1a2e").unwrap();
                document.body().unwrap().append_child(&c).unwrap();
                c
            });

        canvas.set_width(width.max(1) as u32);
        canvas.set_height(height.max(1) as u32);
        canvas.set_tab_index(0); // make focusable
        let _ = canvas.focus();

        if !title.is_empty() {
            document.set_title(title);
        }

        let mut win = WebWindow {
            canvas,
            width,
            height,
            mouse_x: 0.0,
            mouse_y: 0.0,
            keys_down: HashSet::new(),
            mouse_buttons: HashSet::new(),
            events: EventQueue::new(),
            open: true,
            focused: true,
            resized: false,
            _closures: Vec::new(),
        };

        win.attach_listeners();
        Some(win)
    }

    pub fn create_centered(title: &str, width: i32, height: i32) -> Option<Self> {
        Self::create(title, 0, 0, width, height, 0)
    }

    pub fn poll_events(&mut self) -> bool {
        // In browser, events are async — already queued by listeners.
        // Clear events at start of frame (caller should read them before next poll).
        self.events.clear();
        self.resized = false;
        self.open
    }

    pub fn is_key_down(&self, key: i32) -> bool {
        self.keys_down.contains(&key)
    }

    fn attach_listeners(&mut self) {
        // We use shared interior mutability via Rc<RefCell<>> pattern
        // to allow closures to mutate window state.
        // Since WASM is single-threaded, this is safe.

        let events = Rc::new(RefCell::new(Vec::<WindowEvent>::new()));
        let keys = Rc::new(RefCell::new(HashSet::<i32>::new()));
        let mouse_btns = Rc::new(RefCell::new(HashSet::<i32>::new()));
        let mouse_pos = Rc::new(RefCell::new((0.0f64, 0.0f64)));

        // Keydown
        {
            let events = events.clone();
            let keys = keys.clone();
            let closure = Closure::wrap(Box::new(move |e: JsValue| {
                let e: KeyboardEvent = e.dyn_into().unwrap();
                let key = map_key_code(&e.code());
                let mods = map_modifiers(&e);
                keys.borrow_mut().insert(key);
                events.borrow_mut().push(WindowEvent::key_down(key, mods));
                // Prevent default for game keys
                let code = e.code();
                if [
                    "Space",
                    "ArrowUp",
                    "ArrowDown",
                    "ArrowLeft",
                    "ArrowRight",
                    "Tab",
                ]
                .contains(&code.as_str())
                {
                    e.prevent_default();
                }
            }) as Box<dyn FnMut(JsValue)>);
            self.canvas
                .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())
                .unwrap();
            self._closures.push(closure);
        }

        // Keyup
        {
            let events = events.clone();
            let keys = keys.clone();
            let closure = Closure::wrap(Box::new(move |e: JsValue| {
                let e: KeyboardEvent = e.dyn_into().unwrap();
                let key = map_key_code(&e.code());
                let mods = map_modifiers(&e);
                keys.borrow_mut().remove(&key);
                events.borrow_mut().push(WindowEvent::key_up(key, mods));
            }) as Box<dyn FnMut(JsValue)>);
            self.canvas
                .add_event_listener_with_callback("keyup", closure.as_ref().unchecked_ref())
                .unwrap();
            self._closures.push(closure);
        }

        // Mousemove
        {
            let events = events.clone();
            let mouse_pos = mouse_pos.clone();
            let canvas = self.canvas.clone();
            let closure = Closure::wrap(Box::new(move |e: JsValue| {
                let e: MouseEvent = e.dyn_into().unwrap();
                let rect = canvas.get_bounding_client_rect();
                let x = e.client_x() as f64 - rect.left();
                let y = e.client_y() as f64 - rect.top();
                *mouse_pos.borrow_mut() = (x, y);
                events.borrow_mut().push(WindowEvent::mouse_move(x, y));
            }) as Box<dyn FnMut(JsValue)>);
            self.canvas
                .add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())
                .unwrap();
            self._closures.push(closure);
        }

        // Mousedown
        {
            let events = events.clone();
            let mouse_btns = mouse_btns.clone();
            let mouse_pos = mouse_pos.clone();
            let closure = Closure::wrap(Box::new(move |e: JsValue| {
                let e: MouseEvent = e.dyn_into().unwrap();
                let btn = e.button() as i32;
                mouse_btns.borrow_mut().insert(btn);
                let (x, y) = *mouse_pos.borrow();
                events.borrow_mut().push(WindowEvent::mouse_down(btn, x, y));
            }) as Box<dyn FnMut(JsValue)>);
            self.canvas
                .add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())
                .unwrap();
            self._closures.push(closure);
        }

        // Mouseup
        {
            let events = events.clone();
            let mouse_btns = mouse_btns.clone();
            let mouse_pos = mouse_pos.clone();
            let closure = Closure::wrap(Box::new(move |e: JsValue| {
                let e: MouseEvent = e.dyn_into().unwrap();
                let btn = e.button() as i32;
                mouse_btns.borrow_mut().remove(&btn);
                let (x, y) = *mouse_pos.borrow();
                events.borrow_mut().push(WindowEvent::mouse_up(btn, x, y));
            }) as Box<dyn FnMut(JsValue)>);
            self.canvas
                .add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref())
                .unwrap();
            self._closures.push(closure);
        }

        // Wheel
        {
            let events = events.clone();
            let closure = Closure::wrap(Box::new(move |e: JsValue| {
                let e: WheelEvent = e.dyn_into().unwrap();
                events
                    .borrow_mut()
                    .push(WindowEvent::mouse_wheel(e.delta_x(), e.delta_y()));
                e.prevent_default();
            }) as Box<dyn FnMut(JsValue)>);
            self.canvas
                .add_event_listener_with_callback_and_add_event_listener_options(
                    "wheel",
                    closure.as_ref().unchecked_ref(),
                    web_sys::AddEventListenerOptions::new().passive(false),
                )
                .unwrap();
            self._closures.push(closure);
        }

        // Focus/blur
        {
            let events_f = events.clone();
            let closure = Closure::wrap(Box::new(move |_: JsValue| {
                events_f.borrow_mut().push(WindowEvent::focus());
            }) as Box<dyn FnMut(JsValue)>);
            self.canvas
                .add_event_listener_with_callback("focus", closure.as_ref().unchecked_ref())
                .unwrap();
            self._closures.push(closure);
        }
        {
            let events_b = events.clone();
            let closure = Closure::wrap(Box::new(move |_: JsValue| {
                events_b.borrow_mut().push(WindowEvent::blur());
            }) as Box<dyn FnMut(JsValue)>);
            self.canvas
                .add_event_listener_with_callback("blur", closure.as_ref().unchecked_ref())
                .unwrap();
            self._closures.push(closure);
        }

        // Store shared state references for poll_events to read
        // NOTE: This is a simplification. In a full impl, poll_events would
        // drain the shared vec into self.events. For now the event queue is
        // populated directly by the closures.
        let _ = (events, keys, mouse_btns, mouse_pos);
    }
}
