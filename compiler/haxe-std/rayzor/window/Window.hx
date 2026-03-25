package rayzor.window;

/**
 * Native platform window.
 *
 * Creates and manages a native OS window (Cocoa on macOS, X11 on Linux,
 * Win32 on Windows). No third-party dependencies — uses system APIs directly.
 *
 * For GPU rendering, pass getHandle()/getDisplayHandle() to Surface.create().
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
@:native("rayzor::window::Window")
extern class Window {
    // === Creation ===

    /** Create a window with position, size, and style flags (see WindowStyle). */
    @:native("rayzor_window_create")
    public static function create(title:String, x:Int, y:Int, width:Int, height:Int, style:Int):Window;

    /** Create a centered window with default style (titled+closable+resizable+min+max). */
    @:native("rayzor_window_create_centered")
    public static function createCentered(title:String, width:Int, height:Int):Window;

    // === Event Loop ===

    /** Poll events. Returns false when window should close. */
    @:native("rayzor_window_poll_events")
    public function pollEvents():Bool;

    /** Check if a key is currently pressed (use Key constants). */
    @:native("rayzor_window_is_key_down")
    public function isKeyDown(keyCode:Int):Bool;

    // === Native Handles (for GPU surface creation) ===

    /** Native window handle (NSView* / HWND / X11 Window). */
    @:native("rayzor_window_get_handle")
    public function getHandle():rayzor.Ptr<Void>;

    /** Native display handle (null on macOS/Windows, Display* on Linux). */
    @:native("rayzor_window_get_display_handle")
    public function getDisplayHandle():rayzor.Ptr<Void>;

    // === Geometry ===

    @:native("rayzor_window_get_width")
    public function getWidth():Int;

    @:native("rayzor_window_get_height")
    public function getHeight():Int;

    @:native("rayzor_window_get_x")
    public function getX():Int;

    @:native("rayzor_window_get_y")
    public function getY():Int;

    /** True if window was resized since last pollEvents(). */
    @:native("rayzor_window_was_resized")
    public function wasResized():Bool;

    /** Set window position (screen coordinates). */
    @:native("rayzor_window_set_position")
    public function setPosition(x:Int, y:Int):Void;

    /** Set window content area size. */
    @:native("rayzor_window_set_size")
    public function setSize(width:Int, height:Int):Void;

    /** Set minimum window size. */
    @:native("rayzor_window_set_min_size")
    public function setMinSize(width:Int, height:Int):Void;

    /** Set maximum window size. */
    @:native("rayzor_window_set_max_size")
    public function setMaxSize(width:Int, height:Int):Void;

    // === Appearance ===

    @:native("rayzor_window_set_title")
    public function setTitle(title:String):Void;

    @:native("rayzor_window_set_fullscreen")
    public function setFullscreen(fullscreen:Bool):Void;

    @:native("rayzor_window_set_visible")
    public function setVisible(visible:Bool):Void;

    /** Set always-on-top. */
    @:native("rayzor_window_set_floating")
    public function setFloating(alwaysOnTop:Bool):Void;

    /** Set window opacity (0.0 = transparent, 1.0 = opaque). */
    @:native("rayzor_window_set_opacity")
    public function setOpacity(opacity:Float):Void;

    // === State ===

    @:native("rayzor_window_is_fullscreen")
    public function isFullscreen():Bool;

    @:native("rayzor_window_is_visible")
    public function isVisible():Bool;

    @:native("rayzor_window_is_minimized")
    public function isMinimized():Bool;

    @:native("rayzor_window_is_focused")
    public function isFocused():Bool;

    // === Mouse Input ===

    /** Mouse X position relative to window content area. */
    @:native("rayzor_window_get_mouse_x")
    public function getMouseX():Float;

    /** Mouse Y position relative to window content area. */
    @:native("rayzor_window_get_mouse_y")
    public function getMouseY():Float;

    /** Check if a mouse button is pressed (0=left, 1=right, 2=middle). */
    @:native("rayzor_window_is_mouse_down")
    public function isMouseDown(button:Int):Bool;

    // === Cleanup ===

    @:native("rayzor_window_destroy")
    public function destroy():Void;
}
