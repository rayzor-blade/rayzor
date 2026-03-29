package rayzor.window;

/**
 * Platform window — native OS window or browser canvas.
 *
 * On native: Cocoa (macOS), X11 (Linux), Win32 (Windows).
 * On WASM: HTML `<canvas>` element with DOM event listeners.
 *
 * For GPU rendering, pass getHandle()/getDisplayHandle() to Surface.create(),
 * or use Surface.createCanvas() on WASM.
 *
 * Example:
 * ```haxe
 * var win = Window.createCentered("My App", 1280, 720);
 * while (win.pollEvents()) {
 *     if (win.isKeyDown(Key.ESCAPE)) break;
 *     if (win.wasResized()) surface.resize(device, win.getWidth(), win.getHeight());
 *     // render...
 * }
 * win.destroy();
 * ```
 */
#if wasm
@:jsImport("rayzor-window")
#else
@:native("rayzor::window::Window")
#end
extern class Window {
    // === Creation ===

    #if wasm
    @:jsMethod("create")
    public static function create(title:String, x:Int, y:Int, width:Int, height:Int, style:Int):Window;
    @:jsMethod("create-centered")
    public static function createCentered(title:String, width:Int, height:Int):Window;
    #else
    @:native("rayzor_window_create")
    public static function create(title:String, x:Int, y:Int, width:Int, height:Int, style:Int):Window;
    @:native("rayzor_window_create_centered")
    public static function createCentered(title:String, width:Int, height:Int):Window;
    #end

    // === Event Loop ===

    #if wasm
    @:jsMethod("poll-events")
    public function pollEvents():Bool;
    @:jsMethod("is-key-down")
    public function isKeyDown(keyCode:Int):Bool;
    /** Run a frame-driven render loop. The callback is called once per frame
     *  via requestAnimationFrame (browser) or while loop (native).
     *  Return true from the callback to continue, false to stop.
     *  On WASM: non-blocking, yields to browser each frame.
     *  On native: blocking, returns when callback returns false. */
    @:jsMethod("run-loop")
    public static function runLoop(win:Window, callback:Dynamic):Void;
    #else
    @:native("rayzor_window_poll_events")
    public function pollEvents():Bool;
    @:native("rayzor_window_is_key_down")
    public function isKeyDown(keyCode:Int):Bool;
    @:native("rayzor_window_run_loop")
    public static function runLoop(win:Window, callback:Dynamic):Void;
    #end

    // === Native Handles ===

    #if wasm
    @:jsMethod("get-handle")
    public function getHandle():rayzor.Ptr<Void>;
    @:jsMethod("get-display-handle")
    public function getDisplayHandle():rayzor.Ptr<Void>;
    #else
    @:native("rayzor_window_get_handle")
    public function getHandle():rayzor.Ptr<Void>;
    @:native("rayzor_window_get_display_handle")
    public function getDisplayHandle():rayzor.Ptr<Void>;
    #end

    // === Geometry ===

    #if wasm
    @:jsMethod("get-width")
    public function getWidth():Int;
    @:jsMethod("get-height")
    public function getHeight():Int;
    @:jsMethod("get-x")
    public function getX():Int;
    @:jsMethod("get-y")
    public function getY():Int;
    @:jsMethod("was-resized")
    public function wasResized():Bool;
    @:jsMethod("set-position")
    public function setPosition(x:Int, y:Int):Void;
    @:jsMethod("set-size")
    public function setSize(width:Int, height:Int):Void;
    @:jsMethod("set-min-size")
    public function setMinSize(width:Int, height:Int):Void;
    @:jsMethod("set-max-size")
    public function setMaxSize(width:Int, height:Int):Void;
    #else
    @:native("rayzor_window_get_width")
    public function getWidth():Int;
    @:native("rayzor_window_get_height")
    public function getHeight():Int;
    @:native("rayzor_window_get_x")
    public function getX():Int;
    @:native("rayzor_window_get_y")
    public function getY():Int;
    @:native("rayzor_window_was_resized")
    public function wasResized():Bool;
    @:native("rayzor_window_set_position")
    public function setPosition(x:Int, y:Int):Void;
    @:native("rayzor_window_set_size")
    public function setSize(width:Int, height:Int):Void;
    @:native("rayzor_window_set_min_size")
    public function setMinSize(width:Int, height:Int):Void;
    @:native("rayzor_window_set_max_size")
    public function setMaxSize(width:Int, height:Int):Void;
    #end

    // === Appearance ===

    #if wasm
    @:jsMethod("set-title")
    public function setTitle(title:String):Void;
    @:jsMethod("set-fullscreen")
    public function setFullscreen(fullscreen:Bool):Void;
    @:jsMethod("set-visible")
    public function setVisible(visible:Bool):Void;
    @:jsMethod("set-floating")
    public function setFloating(alwaysOnTop:Bool):Void;
    @:jsMethod("set-opacity")
    public function setOpacity(opacity:Float):Void;
    #else
    @:native("rayzor_window_set_title")
    public function setTitle(title:String):Void;
    @:native("rayzor_window_set_fullscreen")
    public function setFullscreen(fullscreen:Bool):Void;
    @:native("rayzor_window_set_visible")
    public function setVisible(visible:Bool):Void;
    @:native("rayzor_window_set_floating")
    public function setFloating(alwaysOnTop:Bool):Void;
    @:native("rayzor_window_set_opacity")
    public function setOpacity(opacity:Float):Void;
    #end

    // === State ===

    #if wasm
    @:jsMethod("is-fullscreen")
    public function isFullscreen():Bool;
    @:jsMethod("is-visible")
    public function isVisible():Bool;
    @:jsMethod("is-minimized")
    public function isMinimized():Bool;
    @:jsMethod("is-focused")
    public function isFocused():Bool;
    #else
    @:native("rayzor_window_is_fullscreen")
    public function isFullscreen():Bool;
    @:native("rayzor_window_is_visible")
    public function isVisible():Bool;
    @:native("rayzor_window_is_minimized")
    public function isMinimized():Bool;
    @:native("rayzor_window_is_focused")
    public function isFocused():Bool;
    #end

    // === Mouse Input ===

    #if wasm
    @:jsMethod("get-mouse-x")
    public function getMouseX():Float;
    @:jsMethod("get-mouse-y")
    public function getMouseY():Float;
    @:jsMethod("is-mouse-down")
    public function isMouseDown(button:Int):Bool;
    #else
    @:native("rayzor_window_get_mouse_x")
    public function getMouseX():Float;
    @:native("rayzor_window_get_mouse_y")
    public function getMouseY():Float;
    @:native("rayzor_window_is_mouse_down")
    public function isMouseDown(button:Int):Bool;
    #end

    // === Event Queue ===

    #if wasm
    @:jsMethod("event-count")
    public function eventCount():Int;
    @:jsMethod("event-type")
    public function eventType(index:Int):Int;
    @:jsMethod("event-x")
    public function eventX(index:Int):Float;
    @:jsMethod("event-y")
    public function eventY(index:Int):Float;
    @:jsMethod("event-key")
    public function eventKey(index:Int):Int;
    @:jsMethod("event-button")
    public function eventButton(index:Int):Int;
    @:jsMethod("event-modifiers")
    public function eventModifiers(index:Int):Int;
    @:jsMethod("event-width")
    public function eventWidth(index:Int):Int;
    @:jsMethod("event-height")
    public function eventHeight(index:Int):Int;
    @:jsMethod("event-scroll-x")
    public function eventScrollX(index:Int):Float;
    @:jsMethod("event-scroll-y")
    public function eventScrollY(index:Int):Float;
    #else
    @:native("rayzor_window_event_count")
    public function eventCount():Int;
    @:native("rayzor_window_event_type")
    public function eventType(index:Int):Int;
    @:native("rayzor_window_event_x")
    public function eventX(index:Int):Float;
    @:native("rayzor_window_event_y")
    public function eventY(index:Int):Float;
    @:native("rayzor_window_event_key")
    public function eventKey(index:Int):Int;
    @:native("rayzor_window_event_button")
    public function eventButton(index:Int):Int;
    @:native("rayzor_window_event_modifiers")
    public function eventModifiers(index:Int):Int;
    @:native("rayzor_window_event_width")
    public function eventWidth(index:Int):Int;
    @:native("rayzor_window_event_height")
    public function eventHeight(index:Int):Int;
    @:native("rayzor_window_event_scroll_x")
    public function eventScrollX(index:Int):Float;
    @:native("rayzor_window_event_scroll_y")
    public function eventScrollY(index:Int):Float;
    #end

    // === Cleanup ===

    #if wasm
    @:jsMethod("destroy")
    public function destroy():Void;
    #else
    @:native("rayzor_window_destroy")
    public function destroy():Void;
    #end
}
