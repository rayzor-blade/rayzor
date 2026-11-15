package p;

// A default the caller's module cannot evaluate is applied in the callee:
// an omitted (or null) enum argument takes its default constructor.
class Pointer {
	public static function press(x:Int, button:MouseButton = Left):String
		return x + " " + Std.string(button) + " " + button.match(Left);

	public static function second(x:Int, button:MouseButton = Right):String
		return x + " " + Std.string(button);

	public static function label(x:Int, text:String = "none"):String
		return x + " " + text;
}
