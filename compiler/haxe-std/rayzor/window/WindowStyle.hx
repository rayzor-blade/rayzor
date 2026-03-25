package rayzor.window;

/** Window style flags (combine with |). */
class WindowStyle {
    public static inline var TITLED:Int      = 1;
    public static inline var CLOSABLE:Int    = 2;
    public static inline var RESIZABLE:Int   = 4;
    public static inline var MINIMIZABLE:Int = 8;
    public static inline var MAXIMIZABLE:Int = 16;
    public static inline var FRAMELESS:Int   = 32;
    public static inline var FLOATING:Int    = 64;
    public static inline var FULLSCREEN:Int  = 128;
    public static inline var DEFAULT:Int     = 1 | 2 | 4 | 8 | 16;
}
