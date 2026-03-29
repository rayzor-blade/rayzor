// Rayzor Window Host — Browser implementation of rayzor:window WIT interface.
//
// Provides DOM canvas-based windowing with keyboard, mouse, and resize events.
// This file is included in the generated JS harness when Window rpkg is detected.

const WINDOW_HOST = (() => {
  let nextHandle = 1;
  const windows = new Map();

  // Key code mapping: rayzor Key constants → DOM key names
  const KEY_MAP = {
    'Escape': 256, 'Enter': 257, 'Tab': 258, 'Backspace': 259,
    'ArrowUp': 265, 'ArrowDown': 264, 'ArrowLeft': 263, 'ArrowRight': 262,
    'Space': 32, 'Delete': 261,
    'ShiftLeft': 340, 'ShiftRight': 340,
    'ControlLeft': 341, 'ControlRight': 341,
    'AltLeft': 342, 'AltRight': 342,
    'MetaLeft': 343, 'MetaRight': 343,
  };
  // A-Z → 65-90
  for (let i = 0; i < 26; i++) {
    KEY_MAP[`Key${String.fromCharCode(65 + i)}`] = 65 + i;
  }
  // 0-9 → 48-57
  for (let i = 0; i < 10; i++) {
    KEY_MAP[`Digit${i}`] = 48 + i;
  }
  // F1-F12 → 290-301
  for (let i = 1; i <= 12; i++) {
    KEY_MAP[`F${i}`] = 289 + i;
  }

  function mapKey(code) { return KEY_MAP[code] ?? 0; }
  function mapModifiers(e) {
    return (e.shiftKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0);
  }

  // Event type constants (matching rayzor EventType)
  const EVT = {
    MOUSE_MOVE: 0, MOUSE_DOWN: 1, MOUSE_UP: 2, MOUSE_WHEEL: 3,
    KEY_DOWN: 4, KEY_UP: 5, RESIZE: 6, MOVE: 7,
    FOCUS: 8, BLUR: 9, CLOSE: 10, MOUSE_ENTER: 11, MOUSE_LEAVE: 12,
  };

  function createWindow(title, x, y, width, height, style) {
    let canvas = document.querySelector('canvas');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'rayzor-window';
      canvas.style.cssText = 'display:block; margin:0 auto; background:#1a1a2e;';
      document.body.appendChild(canvas);
    }
    canvas.width = width || 800;
    canvas.height = height || 600;
    canvas.tabIndex = 0; // Make focusable for keyboard events
    canvas.focus();

    const win = {
      canvas,
      width: canvas.width,
      height: canvas.height,
      x: x || 0,
      y: y || 0,
      mouseX: 0, mouseY: 0,
      keysDown: new Set(),
      mouseButtons: new Set(),
      events: [],
      resized: false,
      open: true,
      focused: true,
    };

    // Event listeners
    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      win.mouseX = e.clientX - rect.left;
      win.mouseY = e.clientY - rect.top;
      win.events.push({ type: EVT.MOUSE_MOVE, x: win.mouseX, y: win.mouseY, key: 0, button: 0, mods: mapModifiers(e), w: 0, h: 0, sx: 0, sy: 0 });
    });
    canvas.addEventListener('mousedown', (e) => {
      win.mouseButtons.add(e.button);
      win.events.push({ type: EVT.MOUSE_DOWN, x: win.mouseX, y: win.mouseY, key: 0, button: e.button, mods: mapModifiers(e), w: 0, h: 0, sx: 0, sy: 0 });
    });
    canvas.addEventListener('mouseup', (e) => {
      win.mouseButtons.delete(e.button);
      win.events.push({ type: EVT.MOUSE_UP, x: win.mouseX, y: win.mouseY, key: 0, button: e.button, mods: mapModifiers(e), w: 0, h: 0, sx: 0, sy: 0 });
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      win.events.push({ type: EVT.MOUSE_WHEEL, x: win.mouseX, y: win.mouseY, key: 0, button: 0, mods: mapModifiers(e), w: 0, h: 0, sx: e.deltaX, sy: e.deltaY });
    }, { passive: false });
    canvas.addEventListener('mouseenter', () => {
      win.events.push({ type: EVT.MOUSE_ENTER, x: win.mouseX, y: win.mouseY, key: 0, button: 0, mods: 0, w: 0, h: 0, sx: 0, sy: 0 });
    });
    canvas.addEventListener('mouseleave', () => {
      win.events.push({ type: EVT.MOUSE_LEAVE, x: win.mouseX, y: win.mouseY, key: 0, button: 0, mods: 0, w: 0, h: 0, sx: 0, sy: 0 });
    });
    document.addEventListener('keydown', (e) => {
      const key = mapKey(e.code);
      win.keysDown.add(key);
      win.events.push({ type: EVT.KEY_DOWN, x: 0, y: 0, key, button: 0, mods: mapModifiers(e), w: 0, h: 0, sx: 0, sy: 0 });
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Tab'].includes(e.key)) e.preventDefault();
    });
    document.addEventListener('keyup', (e) => {
      const key = mapKey(e.code);
      win.keysDown.delete(key);
      win.events.push({ type: EVT.KEY_UP, x: 0, y: 0, key, button: 0, mods: mapModifiers(e), w: 0, h: 0, sx: 0, sy: 0 });
    });
    canvas.addEventListener('focus', () => {
      win.focused = true;
      win.events.push({ type: EVT.FOCUS, x: 0, y: 0, key: 0, button: 0, mods: 0, w: 0, h: 0, sx: 0, sy: 0 });
    });
    canvas.addEventListener('blur', () => {
      win.focused = false;
      win.events.push({ type: EVT.BLUR, x: 0, y: 0, key: 0, button: 0, mods: 0, w: 0, h: 0, sx: 0, sy: 0 });
    });

    // ResizeObserver for canvas size changes
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width: w, height: h } = entry.contentRect;
          if (w !== win.width || h !== win.height) {
            win.width = Math.round(w);
            win.height = Math.round(h);
            canvas.width = win.width;
            canvas.height = win.height;
            win.resized = true;
            win.events.push({ type: EVT.RESIZE, x: 0, y: 0, key: 0, button: 0, mods: 0, w: win.width, h: win.height, sx: 0, sy: 0 });
          }
        }
      });
      observer.observe(canvas);
      win._observer = observer;
    }

    const h = nextHandle++;
    windows.set(h, win);
    return h;
  }

  return {
    'create': (title, x, y, width, height, style) => createWindow(title, x, y, width, height, style),
    'create-centered': (title, width, height) => createWindow(title, 0, 0, width, height, 0),
    'destroy': (win) => {
      const w = windows.get(win);
      if (w?._observer) w._observer.disconnect();
      windows.delete(win);
    },

    'poll-events': (win) => {
      const w = windows.get(win);
      if (!w) return 0;
      // In browser, events are async — they've already been queued by listeners.
      // Just return whether window is still open.
      const wasResized = w.resized;
      w.resized = false;
      return w.open ? 1 : 0;
    },

    'get-width': (win) => windows.get(win)?.width ?? 0,
    'get-height': (win) => windows.get(win)?.height ?? 0,
    'get-x': (win) => windows.get(win)?.x ?? 0,
    'get-y': (win) => windows.get(win)?.y ?? 0,
    'was-resized': (win) => windows.get(win)?.resized ? 1 : 0,
    'set-position': (win, x, y) => { const w = windows.get(win); if (w) { w.x = x; w.y = y; } },
    'set-size': (win, width, height) => {
      const w = windows.get(win);
      if (w) { w.canvas.width = width; w.canvas.height = height; w.width = width; w.height = height; }
    },
    'set-min-size': () => {},
    'set-max-size': () => {},
    'set-title': (win, title) => { document.title = title?.toString() ?? ''; },
    'set-fullscreen': (win, fs) => {
      const w = windows.get(win);
      if (w && fs) w.canvas.requestFullscreen?.();
      else document.exitFullscreen?.();
    },
    'set-visible': (win, visible) => {
      const w = windows.get(win);
      if (w) w.canvas.style.display = visible ? 'block' : 'none';
    },
    'set-floating': () => {},
    'set-opacity': (win, opacity) => {
      const w = windows.get(win);
      if (w) w.canvas.style.opacity = `${opacity}`;
    },
    'is-fullscreen': () => document.fullscreenElement ? 1 : 0,
    'is-visible': (win) => windows.get(win)?.canvas.style.display !== 'none' ? 1 : 0,
    'is-minimized': () => document.hidden ? 1 : 0,
    'is-focused': (win) => windows.get(win)?.focused ? 1 : 0,
    'is-key-down': (win, keyCode) => windows.get(win)?.keysDown.has(keyCode) ? 1 : 0,
    'get-mouse-x': (win) => windows.get(win)?.mouseX ?? 0.0,
    'get-mouse-y': (win) => windows.get(win)?.mouseY ?? 0.0,
    'is-mouse-down': (win, button) => windows.get(win)?.mouseButtons.has(button) ? 1 : 0,
    'get-handle': (win) => win, // Canvas handle IS the window handle
    'get-display-handle': () => 0,

    // Event queue
    'event-count': (win) => windows.get(win)?.events.length ?? 0,
    'event-type': (win, i) => windows.get(win)?.events[i]?.type ?? 0,
    'event-x': (win, i) => windows.get(win)?.events[i]?.x ?? 0.0,
    'event-y': (win, i) => windows.get(win)?.events[i]?.y ?? 0.0,
    'event-key': (win, i) => windows.get(win)?.events[i]?.key ?? 0,
    'event-button': (win, i) => windows.get(win)?.events[i]?.button ?? 0,
    'event-modifiers': (win, i) => windows.get(win)?.events[i]?.mods ?? 0,
    'event-width': (win, i) => windows.get(win)?.events[i]?.w ?? 0,
    'event-height': (win, i) => windows.get(win)?.events[i]?.h ?? 0,
    'event-scroll-x': (win, i) => windows.get(win)?.events[i]?.sx ?? 0.0,
    'event-scroll-y': (win, i) => windows.get(win)?.events[i]?.sy ?? 0.0,

    // Drain events — call at end of frame
    _drainEvents(win) {
      const w = windows.get(win);
      if (w) w.events.length = 0;
    },
  };
})();

export default WINDOW_HOST;
