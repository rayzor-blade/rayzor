package rayzor.window;

/** Window event types returned by Window.eventType(). */
class EventType {
    public static inline var NONE:Int        = 0;
    public static inline var MOUSE_MOVE:Int  = 1;
    public static inline var MOUSE_DOWN:Int  = 2;
    public static inline var MOUSE_UP:Int    = 3;
    public static inline var MOUSE_WHEEL:Int = 4;
    public static inline var KEY_DOWN:Int    = 5;
    public static inline var KEY_UP:Int      = 6;
    public static inline var RESIZE:Int      = 7;
    public static inline var MOVE:Int        = 8;
    public static inline var FOCUS:Int       = 9;
    public static inline var BLUR:Int        = 10;
    public static inline var CLOSE:Int       = 11;
    public static inline var MOUSE_ENTER:Int = 12;
    public static inline var MOUSE_LEAVE:Int = 13;
}
