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
 *     // render...
 * }
 * win.destroy();
 * ```
 */
@:native("rayzor::window::Window")
extern class Window {
    /** Create a window with position, size, and style flags. */
    @:native("rayzor_window_create")
    public static function create(title:String, x:Int, y:Int, width:Int, height:Int, style:Int):Window;

    /** Create a centered window with default style. */
    @:native("rayzor_window_create_centered")
    public static function createCentered(title:String, width:Int, height:Int):Window;

    /** Poll events. Returns false when window should close. */
    @:native("rayzor_window_poll_events")
    public function pollEvents():Bool;

    /** Check if a key is currently pressed (use Key constants). */
    @:native("rayzor_window_is_key_down")
    public function isKeyDown(keyCode:Int):Bool;

    /** Native window handle for GPU surface creation (NSView*/HWND/X11 Window). */
    @:native("rayzor_window_get_handle")
    public function getHandle():rayzor.Ptr<Void>;

    /** Native display handle (null on macOS/Windows, Display* on Linux). */
    @:native("rayzor_window_get_display_handle")
    public function getDisplayHandle():rayzor.Ptr<Void>;

    @:native("rayzor_window_get_width")
    public function getWidth():Int;

    @:native("rayzor_window_get_height")
    public function getHeight():Int;

    /** True if window was resized since last pollEvents(). */
    @:native("rayzor_window_was_resized")
    public function wasResized():Bool;

    @:native("rayzor_window_set_title")
    public function setTitle(title:String):Void;

    @:native("rayzor_window_destroy")
    public function destroy():Void;
}
